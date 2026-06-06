"""OM1 MCP server: expose OM1 tools to external MCP clients."""

from __future__ import annotations

import asyncio
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from expose.config import ServerConfig
from expose.tools import Emotion, MoveAction, build_tool_definitions
from expose.websim_adapter import WebSimAdapter

logger = logging.getLogger(__name__)


def _get_version() -> str:
    try:
        return version("om1")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _err(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=f"Error: {msg}")]


def _ok(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=msg)]


async def handle_tool_call(
    name: str, arguments: dict[str, Any] | None, adapter: WebSimAdapter
) -> list[types.TextContent]:
    """Dispatch an MCP tool call to the adapter and return the tool result as TextContent.

    Invalid inputs return an error TextContent instead of raising, so the MCP
    client observes a structured failure rather than a transport-level crash.
    """
    args = arguments or {}
    try:
        if name == "om1_move":
            action = MoveAction(args["action"])
            adapter.move(action.value)
            return _ok(f"Executed move: {action.value}")

        if name == "om1_speak":
            text = args.get("text", "").strip()
            if not text:
                raise ValueError("text must be non-empty")
            adapter.speak(text)
            return _ok(f"Spoke: {text}")

        if name == "om1_face":
            emotion = Emotion(args["emotion"])
            adapter.face(emotion.value)
            return _ok(f"Changed emotion to: {emotion.value}")

        return _err(f"Unknown tool: {name}")
    except (KeyError, ValueError) as e:
        return _err(str(e))


def build_server(adapter: WebSimAdapter) -> Server:
    """Wire an mcp.server.Server with list_tools/call_tool handlers backed by ``adapter``."""
    server = Server("om1_mcp_server")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return build_tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        return await handle_tool_call(name, arguments, adapter)

    return server


async def run(config: ServerConfig) -> None:
    """Initialise logging/WebSim and serve the MCP protocol over stdio until stdin closes."""
    logging.basicConfig(level=config.log_level.upper())
    adapter = WebSimAdapter.create(config.websim_host, config.websim_port)
    server = build_server(adapter)

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="om1_mcp_server",
                server_version=_get_version(),
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    """CLI entry point (``om1-mcp-server``): read env config and run the server."""
    asyncio.run(run(ServerConfig.from_env()))


if __name__ == "__main__":
    main()
