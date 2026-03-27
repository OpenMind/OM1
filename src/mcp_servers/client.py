import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from pydantic import BaseModel


class StdioServerConfig(BaseModel):
    """Configuration for an MCP server using stdio transport."""

    name: str
    command: str
    args: List[str] = []
    env: Optional[Dict[str, str]] = None


@dataclass
class MCPTool:
    """Metadata for a single MCP tool."""

    key: str
    server_name: str
    original_name: str
    description: str
    input_schema: dict

    def convert_to_schema(self) -> dict:
        """Convert to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.key,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def generate_description(self) -> str:
        """Generate description for LLM prompts."""
        params = self.input_schema.get("properties", {})
        param_str = ", ".join(
            f"{param_name}: {param_info.get('type', 'any')}"
            for param_name, param_info in params.items()
        )
        return f"- {self.key}({param_str}): {self.description}"


class StdioTransport:
    """Handles stdio transport connections to MCP servers."""

    @staticmethod
    async def connect(
        exit_stack: AsyncExitStack, config: StdioServerConfig
    ) -> Tuple[Any, Any]:
        """Open a stdio connection to an MCP server."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
        )
        client_cm = stdio_client(server_params)
        read, write = await exit_stack.enter_async_context(client_cm)
        return read, write


class MCPClientManager:
    """Manage connections to multiple MCP servers.

    Parameters
    ----------
    server_configs : list[dict]
        Raw configuration dicts.
    """

    def __init__(self, server_configs: List[Dict]) -> None:
        self._configs = [StdioServerConfig(**c) for c in server_configs]
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._exit_stack: Optional[AsyncExitStack] = None

        self._connect_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Connect to all configured MCP servers."""
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run_event_loop())
        self._connect_event.set()
        await self._ready.wait()

    async def stop(self) -> None:
        """Disconnect all MCP servers."""
        if not self._ready.is_set():
            return
        self._close_event.set()
        await self._closed.wait()

        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None

    async def _run_event_loop(self) -> None:
        """Internal loop that owns all MCP connections in a single task."""
        try:
            while True:
                await self._connect_event.wait()
                self._connect_event.clear()
                self._closed.clear()

                await self._connect_all()
                self._ready.set()

                await self._close_event.wait()
                self._close_event.clear()
                self._ready.clear()

                await self._close_all()
                self._closed.set()
        except asyncio.CancelledError:
            await self._close_all()
            raise

    async def _connect_all(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for config in self._configs:
            try:
                await self._connect_server(config)
            except Exception as e:
                logging.error(f"Failed to connect to MCP server '{config.name}': {e}")

        logging.info(f"MCP client connected with {len(self._tools)} tools")

    async def _close_all(self) -> None:
        """Close all MCP server connections."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logging.error(f"Error closing MCP connections: {e}")
            self._exit_stack = None
        self._sessions.clear()
        self._tools.clear()

    async def _connect_server(self, config: StdioServerConfig) -> None:
        """Connect to a single MCP server."""
        assert self._exit_stack is not None
        read, write = await StdioTransport.connect(self._exit_stack, config)

        session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(session)
        await session.initialize()

        tools_result = await session.list_tools()
        self._sessions[config.name] = session

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

    def get_tool_schemas(self) -> List[Dict]:
        """Get OpenAI-format function schemas for all MCP tools."""
        return [tool.convert_to_schema() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> str:
        """Get text descriptions of MCP tools for the LLM prompt."""
        if not self._tools:
            return ""
        header = (
            "[MCP Tools]\n"
            "Use these tools to get external information. "
            "Call the tool first, then use the result to respond.\n"
        )
        tool_lines = "\n".join(
            tool.generate_description() for tool in self._tools.values()
        )
        return f"{header}{tool_lines}"

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to an MCP server."""
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
