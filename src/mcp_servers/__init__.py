from typing import Dict, List

from mcp_servers.client import MCPClientManager

__all__ = ["MCPClientManager", "load_mcp"]


def load_mcp(server_configs: List[Dict]) -> MCPClientManager:
    """Create an MCP client manager.


    Parameters
    ----------
    server_configs : list[dict]
        MCP server configurations from config file.

    Returns
    -------
    MCPClientManager
        MCP client manager.
    """
    return MCPClientManager(server_configs or [])
