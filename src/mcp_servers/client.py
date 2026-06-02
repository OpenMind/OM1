import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from pydantic import BaseModel


class TransportType(str, Enum):
    """Supported MCP server transport protocols."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    transport: TransportType = TransportType.STDIO
    command: Optional[str] = None
    args: List[str] = []
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class MCPTool:
    """Metadata for a single MCP tool."""

    key: str
    server_name: str
    original_name: str
    description: str
    input_schema: dict

    def convert_to_schema(self) -> dict:
        """Convert to OpenAI function-calling schema.

        Returns
        -------
        dict
            Dictionary matching the OpenAI function format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.key,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def generate_description(self) -> str:
        """Generate description for LLM prompts.

        Returns
        -------
        str
            A formatted string describing the tool signature and purpose.
        """
        params = self.input_schema.get("properties", {})
        param_str = ", ".join(
            f"{param_name}: {param_info.get('type', 'any')}" for param_name, param_info in params.items()
        )
        return f"- {self.key}({param_str}): {self.description}"


class MCPClientManager:
    """Manage connections to multiple MCP servers.

    Parameters
    ----------
    server_configs : list[dict]
        Raw configuration dicts.
    """

    def __init__(self, server_configs: List[Dict]) -> None:
        """Initialize the MCPClientManager with server configurations."""
        self._configs = [MCPServerConfig(**c) for c in server_configs]
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, MCPTool] = {}

        self._close_event: Optional[asyncio.Event] = None
        self._tasks: List[asyncio.Task] = []
        self._started = False

    async def start(self) -> None:
        """Connect to all configured MCP servers.

        This method spawns independent background tasks for each server
        and waits until all connections are either established or failed.
        """
        if self._started:
            return

        if self._close_event is None:
            self._close_event = asyncio.Event()
        self._close_event.clear()

        init_events = []
        for config in self._configs:
            ready_event = asyncio.Event()
            init_events.append(ready_event)
            task = asyncio.create_task(self._run_server_task(config, ready_event))
            self._tasks.append(task)

        if init_events:
            await asyncio.gather(*(e.wait() for e in init_events))

        self._started = True

    async def stop(self) -> None:
        """Disconnect all MCP servers.

        This method signals all server tasks to close and waits for teardown.
        """
        if not self._started:
            return

        if self._close_event:
            self._close_event.set()

        if self._tasks:
            done, pending = await asyncio.wait(self._tasks, timeout=5.0)
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            self._tasks.clear()

        self._sessions.clear()
        self._tools.clear()
        self._started = False

    async def _run_server_task(self, config: MCPServerConfig, ready_event: asyncio.Event) -> None:
        """Independent event loop task for a single MCP server."""
        exit_stack = AsyncExitStack()
        try:
            await exit_stack.__aenter__()
            session = await self._connect_server(config, exit_stack)
            self._sessions[config.name] = session

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                mcp_tool = MCPTool(
                    key=f"mcp_{config.name}_{tool.name}",
                    server_name=config.name,
                    original_name=tool.name,
                    description=tool.description or f"MCP tool: {tool.name}",
                    input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                )
                self._tools[mcp_tool.key] = mcp_tool

            logging.info(
                f"MCP server '{config.name}': {len(tools_result.tools)} tools "
                f"({[t.name for t in tools_result.tools]})"
            )

            ready_event.set()

            if self._close_event:
                await self._close_event.wait()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"MCP server '{config.name}' disconnected unexpectedly: {e}")
        finally:
            ready_event.set()
            try:
                await exit_stack.aclose()
            except Exception as e:
                logging.error(f"Error closing MCP connection '{config.name}': {e}")

            if config.name in self._sessions:
                del self._sessions[config.name]

            keys_to_remove = [k for k, v in self._tools.items() if v.server_name == config.name]
            for k in keys_to_remove:
                del self._tools[k]

    async def _connect_server(self, config: MCPServerConfig, exit_stack: AsyncExitStack) -> ClientSession:
        """Establish the connection boundaries for a specific transport."""
        match config.transport:
            case TransportType.STDIO:
                if not config.command:
                    raise ValueError(f"MCP server '{config.name}' requires 'command' for stdio transport")
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                client_cm = stdio_client(server_params)
                read, write = await exit_stack.enter_async_context(client_cm)

            case TransportType.SSE:
                if not config.url:
                    raise ValueError(f"MCP server '{config.name}' requires 'url' for sse transport")
                client_cm = sse_client(config.url, headers=config.headers)
                read, write = await exit_stack.enter_async_context(client_cm)

            case TransportType.HTTP:
                if not config.url:
                    raise ValueError(f"MCP server '{config.name}' requires 'url' for http transport")
                http_client = httpx.AsyncClient(
                    headers=config.headers or {},
                    timeout=httpx.Timeout(30.0, read=300.0),
                )
                await exit_stack.enter_async_context(http_client)
                client_cm = streamable_http_client(config.url, http_client=http_client)
                read, write, _ = await exit_stack.enter_async_context(client_cm)

            case _:
                raise ValueError(f"Unsupported transport type: {config.transport}")

        session = ClientSession(read, write)
        await exit_stack.enter_async_context(session)
        await session.initialize()
        return session

    def get_tool_schemas(self) -> List[Dict]:
        """Get OpenAI-format function schemas for all MCP tools.

        Returns
        -------
        List[Dict]
            A list of dictionary schemas mapping to registered MCP tools.
        """
        return [tool.convert_to_schema() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> str:
        """Get text descriptions of MCP tools for the LLM prompt.

        Returns
        -------
        str
            A concatenated string block summarizing the tools.
        """
        if not self._tools:
            return ""
        header = (
            "[MCP Tools]\n"
            "Use these tools to get external information. "
            "Call the tool first, then use the result to respond.\n"
        )
        tool_lines = "\n".join(tool.generate_description() for tool in self._tools.values())
        return f"{header}{tool_lines}"

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to an MCP server.

        Parameters
        ----------
        tool_name : str
            The name of the tool to verify.

        Returns
        -------
        bool
            True if the tool is an MCP tool, False otherwise.
        """
        return tool_name in self._tools

    async def call_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the text result.

        Parameters
        ----------
        tool_key : str
            Tool key in format 'mcp_{server}_{tool_name}'.
        arguments : dict
            Arguments to pass to the MCP tool.

        Returns
        -------
        str
            Text result from the tool.
        """
        tool = self._tools.get(tool_key)
        if not tool:
            raise ValueError(f"Unknown MCP tool: {tool_key}")

        session = self._sessions[tool.server_name]
        result = await session.call_tool(tool.original_name, arguments=arguments)

        texts = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)

        return "\n".join(texts) if texts else str(result.content)
