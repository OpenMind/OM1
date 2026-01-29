import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from llm.function_schemas import generate_function_schema_from_action
from llm.output_model import Action
from providers.io_provider import IOProvider

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_name(value: str) -> str:
    """Sanitize names to OpenAI tool name constraints."""
    return _SAFE_NAME_RE.sub("_", value.strip())


def _make_tool_name(prefix: str, server_name: str, tool_name: str) -> str:
    """Build a stable, safe tool name for LLM tool calling."""
    base = f"{prefix}__{_sanitize_name(server_name)}__{_sanitize_name(tool_name)}"
    if len(base) <= 64:
        return base

    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    truncated = base[: 64 - 9]
    return f"{truncated}_{digest}"


def _format_tool_result(result: Any) -> str:
    """Format MCP tool result content into a readable string."""
    if result is None:
        return ""

    # mcp CallToolResult.content is typically a list of content blocks
    content = getattr(result, "content", None)
    if content is None:
        return str(result)

    parts: List[str] = []
    for item in content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        elif hasattr(item, "text"):
            parts.append(getattr(item, "text"))
        else:
            parts.append(str(item))
    return "\n".join([p for p in parts if p])


def _parse_action_arguments(action: Action) -> Dict[str, Any]:
    """Parse Action.value into MCP tool arguments."""
    if not action.value:
        return {}

    try:
        parsed = json.loads(action.value)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"value": action.value}


@dataclass
class MCPClientConfig:
    """Configuration for a single MCP client connection."""

    name: str
    transport: str = "stdio"  # stdio | streamable-http
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    timeout: float = 30.0
    tool_allowlist: List[str] = field(default_factory=list)
    tool_denylist: List[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """Configuration for exposing OM1 as an MCP server."""

    enabled: bool = True
    name: str = "om1"
    transport: str = "streamable-http"  # streamable-http | stdio
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/mcp"
    stateless_http: bool = True
    json_response: bool = True
    allow_actions: List[str] = field(default_factory=list)
    deny_actions: List[str] = field(default_factory=list)


@dataclass
class MCPConfig:
    """Top-level MCP configuration container."""

    enabled: bool = True
    tool_prefix: str = "mcp"
    inject_results_input: bool = True
    results_input_config: Dict[str, Any] = field(default_factory=dict)
    clients: List[MCPClientConfig] = field(default_factory=list)
    server: Optional[MCPServerConfig] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "MCPConfig":
        if not raw:
            return cls(enabled=False)
        if not isinstance(raw, Mapping):
            logging.warning("MCP config is not a mapping; disabling MCP")
            return cls(enabled=False)

        enabled = bool(raw.get("enabled", True))
        tool_prefix = raw.get("tool_prefix", "mcp")
        if tool_prefix != "mcp":
            logging.warning(
                "MCP tool_prefix must be 'mcp' for correct tool parsing; overriding."
            )
            tool_prefix = "mcp"
        inject_results_input = bool(raw.get("inject_results_input", True))
        results_input_config = raw.get("results_input_config", {}) or {}

        clients_cfg = []
        for entry in raw.get("clients", []) or []:
            if not entry or not entry.get("name"):
                continue
            clients_cfg.append(
                MCPClientConfig(
                    name=entry.get("name"),
                    transport=entry.get("transport", "stdio"),
                    command=entry.get("command"),
                    args=entry.get("args", []) or [],
                    env=entry.get("env", {}) or {},
                    url=entry.get("url"),
                    timeout=float(entry.get("timeout", 30.0)),
                    tool_allowlist=entry.get("tool_allowlist", []) or [],
                    tool_denylist=entry.get("tool_denylist", []) or [],
                )
            )

        server_cfg = None
        server_raw = raw.get("server")
        if server_raw:
            server_cfg = MCPServerConfig(
                enabled=bool(server_raw.get("enabled", True)),
                name=server_raw.get("name", "om1"),
                transport=server_raw.get("transport", "streamable-http"),
                host=server_raw.get("host", "0.0.0.0"),
                port=int(server_raw.get("port", 8765)),
                path=server_raw.get("path", "/mcp"),
                stateless_http=bool(server_raw.get("stateless_http", True)),
                json_response=bool(server_raw.get("json_response", True)),
                allow_actions=server_raw.get("allow_actions", []) or [],
                deny_actions=server_raw.get("deny_actions", []) or [],
            )

        return cls(
            enabled=enabled,
            tool_prefix=tool_prefix,
            inject_results_input=inject_results_input,
            results_input_config=results_input_config,
            clients=clients_cfg,
            server=server_cfg,
        )


class MCPClientManager:
    """Manages outbound MCP client sessions and tool registry."""

    def __init__(self, config: MCPConfig):
        self._config = config
        self._sessions: Dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()
        self._tool_map: Dict[str, Tuple[str, str]] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
        self._io_provider = IOProvider()
        self._started = False

    @property
    def tool_names(self) -> List[str]:
        return list(self._tool_map.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(self._tool_schemas)

    async def start(self) -> None:
        if not self._config.enabled:
            return
        if self._started:
            return

        for server in self._config.clients:
            try:
                session = await self._connect(server)
                if session is None:
                    continue
                tools_result = await session.list_tools()
                tools = getattr(tools_result, "tools", [])

                for tool in tools:
                    if not self._tool_allowed(tool.name, server):
                        continue
                    input_schema = getattr(tool, "inputSchema", None) or getattr(
                        tool, "input_schema", None
                    )
                    tool_name = _make_tool_name(
                        self._config.tool_prefix, server.name, tool.name
                    )
                    self._tool_map[tool_name] = (server.name, tool.name)
                    self._tool_schemas.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool.description
                                or f"MCP tool {tool.name}",
                                "parameters": input_schema or {"type": "object"},
                                "strict": True,
                            },
                        }
                    )

                logging.info(
                    f"MCP client '{server.name}' loaded {len(tools)} tools"
                )
            except Exception as e:
                logging.error(f"Failed to connect MCP server {server.name}: {e}")

        self._started = True

    async def stop(self) -> None:
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logging.error(f"Failed to close MCP sessions: {e}")
        finally:
            self._sessions = {}
            self._tool_map = {}
            self._tool_schemas = []
            self._started = False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self._tool_map:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        server_name, remote_tool = self._tool_map[tool_name]
        session = self._sessions.get(server_name)
        if session is None:
            raise ValueError(f"No MCP session for server: {server_name}")

        result = await session.call_tool(remote_tool, arguments=arguments)
        return _format_tool_result(result)

    async def handle_actions(self, actions: List[Action]) -> None:
        if not actions:
            return

        async def _run(action: Action) -> None:
            args = _parse_action_arguments(action)
            try:
                output = await self.call_tool(action.type, args)
                self._record_result(action.type, output, error=None)
            except Exception as e:
                logging.error(f"MCP tool call failed {action.type}: {e}")
                self._record_result(action.type, "", error=str(e))

        await asyncio.gather(*[_run(action) for action in actions])

    def _record_result(self, tool_name: str, output: str, error: Optional[str]):
        entry = {
            "tool": tool_name,
            "output": output,
            "error": error,
            "timestamp": time.time(),
        }
        results = self._io_provider.get_dynamic_variable("mcp_results") or []
        if not isinstance(results, list):
            results = []
        results.append(entry)
        self._io_provider.add_dynamic_variable("mcp_results", results)

    async def _connect(self, server: MCPClientConfig) -> Optional[ClientSession]:
        if server.transport == "stdio":
            if not server.command:
                logging.error(
                    f"MCP stdio server '{server.name}' missing command"
                )
                return None
            env = os.environ.copy()
            env.update(server.env or {})
            params = StdioServerParameters(
                command=server.command,
                args=server.args,
                env=env,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
        elif server.transport == "streamable-http":
            if not server.url:
                logging.error(
                    f"MCP streamable-http server '{server.name}' missing url"
                )
                return None
            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(server.url)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
        else:
            logging.error(
                f"Unknown MCP transport '{server.transport}' for {server.name}"
            )
            return None

        await session.initialize()
        self._sessions[server.name] = session
        return session

    def _tool_allowed(self, name: str, server: MCPClientConfig) -> bool:
        if server.tool_allowlist and name not in server.tool_allowlist:
            return False
        if server.tool_denylist and name in server.tool_denylist:
            return False
        return True


class MCPServerManager:
    """Expose OM1 actions over MCP for external clients."""

    def __init__(self, config: MCPServerConfig):
        self._config = config
        try:
            self._mcp = FastMCP(
                config.name,
                stateless_http=config.stateless_http,
                json_response=config.json_response,
            )
        except TypeError:
            self._mcp = FastMCP(config.name)
        self._actions: List[Any] = []
        self._action_orchestrator = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None

        self._register_tools()

    def update_actions(self, actions: List[Any]) -> None:
        self._actions = actions

    def update_action_orchestrator(self, action_orchestrator: Any) -> None:
        self._action_orchestrator = action_orchestrator

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._event_loop = loop

    def start(self) -> None:
        if not self._config.enabled:
            return
        if self._server_thread and self._server_thread.is_alive():
            return

        def _run():
            try:
                settings = getattr(self._mcp, "settings", None)
                if settings is not None:
                    if hasattr(settings, "host"):
                        settings.host = self._config.host
                    if hasattr(settings, "port"):
                        settings.port = self._config.port
                    if hasattr(settings, "streamable_http_path"):
                        settings.streamable_http_path = self._config.path
                self._mcp.run(transport=self._config.transport)
            except Exception as e:
                logging.error(f"MCP server failed: {e}")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        logging.info(
            f"MCP server started on {self._config.host}:{self._config.port}{self._config.path}"
        )

    def _register_tools(self) -> None:
        @self._mcp.tool()
        async def om1_actions() -> Dict[str, Any]:
            """List available OM1 actions with input schemas."""
            return {
                "actions": [self._describe_action(action) for action in self._actions]
            }

        @self._mcp.tool()
        async def om1_action(action: str, params: Optional[Dict[str, Any]] = None) -> str:
            """Execute an OM1 action by name with optional params."""
            if not self._action_orchestrator:
                return "Action orchestrator not ready"

            action_label = action.strip().lower()
            if not self._is_action_allowed(action_label):
                return f"Action not allowed: {action_label}"

            known = {a.llm_label.lower() for a in self._actions}
            if action_label not in known:
                return f"Unknown action: {action_label}"

            payload = params or {}
            action_value = json.dumps(payload)
            coro = self._action_orchestrator.promise(
                [Action(type=action_label, value=action_value)]
            )
            if self._event_loop and self._event_loop.is_running():
                current_loop = asyncio.get_running_loop()
                if current_loop != self._event_loop:
                    future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
                    await asyncio.wrap_future(future)
                else:
                    await coro
            else:
                await coro
            return "queued"

    def _is_action_allowed(self, action: str) -> bool:
        if self._config.allow_actions and action not in self._config.allow_actions:
            return False
        if self._config.deny_actions and action in self._config.deny_actions:
            return False
        return True

    def _describe_action(self, action: Any) -> Dict[str, Any]:
        schema = generate_function_schema_from_action(action)
        return {
            "name": action.llm_label,
            "description": schema.get("function", {}).get("description", ""),
            "input_schema": schema.get("function", {}).get("parameters", {}),
        }


class MCPBridge:
    """Coordinates MCP client and server integration in OM1 runtime."""

    def __init__(self, raw_config: Optional[Dict[str, Any]]):
        self.config = MCPConfig.from_dict(raw_config)
        self.client_manager: Optional[MCPClientManager] = None
        self.server_manager: Optional[MCPServerManager] = None

        if self.config.enabled:
            if self.config.clients:
                self.client_manager = MCPClientManager(self.config)
            if self.config.server and self.config.server.enabled:
                self.server_manager = MCPServerManager(self.config.server)

    def __bool__(self) -> bool:
        return bool(self.config.enabled)

    async def start(self) -> None:
        if self.client_manager:
            await self.client_manager.start()
        if self.server_manager:
            self.server_manager.start()

    async def stop(self) -> None:
        if self.client_manager:
            await self.client_manager.stop()

    def attach_llm(self, llm: Any) -> None:
        if not llm or not self.client_manager:
            return
        llm.set_extra_function_schemas(self.client_manager.get_tool_schemas())

    def update_actions(self, actions: List[Any]) -> None:
        if self.server_manager:
            self.server_manager.update_actions(actions)

    def update_action_orchestrator(self, action_orchestrator: Any) -> None:
        if self.server_manager:
            self.server_manager.update_action_orchestrator(action_orchestrator)

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        if self.server_manager:
            self.server_manager.set_event_loop(loop)

    def split_actions(self, actions: List[Action]) -> Tuple[List[Action], List[Action]]:
        if not actions or not self.client_manager:
            return [], actions

        mcp_actions = []
        native_actions = []
        tool_names = set(self.client_manager.tool_names)
        for action in actions:
            if action.type in tool_names:
                mcp_actions.append(action)
            else:
                native_actions.append(action)
        return mcp_actions, native_actions

    async def handle_mcp_actions(self, actions: List[Action]) -> None:
        if self.client_manager:
            await self.client_manager.handle_actions(actions)
