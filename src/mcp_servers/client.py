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
        return (
            f"MCP TOOL: {self.key}({param_str})\n"
            f"Description: {self.description}\n"
            f"Use this tool when you need to get external information. "
            f"Call it first, then use the result to respond.\n"
        )


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
    """Manage connections to multiple MCP servers and execute tool calls.

    Parameters
    ----------
    server_configs : list of dict
        Raw configuration dicts.
    """

    def __init__(self, server_configs: List[Dict]) -> None:
        self._configs = [StdioServerConfig(**c) for c in server_configs]
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._exit_stack: Optional[AsyncExitStack] = None

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for config in self._configs:
            try:
                await self._connect_server(config)
            except Exception as e:
                logging.error(f"Failed to connect to MCP server '{config.name}': {e}")

    async def _connect_server(self, config: StdioServerConfig) -> None:
        """Connect to a single MCP server."""
        assert self._exit_stack is not None
        read, write = await StdioTransport.connect(self._exit_stack, config)

        session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(session)
        await session.initialize()

        # Discover tools
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
        return "\n".join(tool.generate_description() for tool in self._tools.values())

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

    async def close_all(self) -> None:
        """Close all MCP server connections."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logging.error(f"Error closing MCP connections: {e}")
            self._exit_stack = None
        self._sessions.clear()
        self._tools.clear()
