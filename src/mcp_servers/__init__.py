import asyncio
import logging
from typing import Dict, List

import nest_asyncio

from mcp_servers.client import MCPClientManager

__all__ = ["MCPClientManager", "load_mcp"]


def load_mcp(server_configs: List[Dict]) -> MCPClientManager:
    """Load and connect MCP servers.

    Parameters
    ----------
    server_configs : list[dict]
        MCP server configurations from config file.

    Returns
    -------
    MCPClientManager
        Connected MCP client.
    """
    if not server_configs:
        return MCPClientManager([])

    try:
        client = MCPClientManager(server_configs)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()
            loop.run_until_complete(client.connect_all())
        else:
            asyncio.run(client.connect_all())
        logging.info(
            f"MCP client initialized with " f"{len(client.get_tool_schemas())} tools"
        )
        return client
    except Exception as e:
        logging.error(f"Failed to initialize MCP client: {e}")
        return MCPClientManager([])
