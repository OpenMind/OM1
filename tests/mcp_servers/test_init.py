from mcp_servers import load_mcp
from mcp_servers.client import MCPClientManager


class TestLoadMcp:
    """Test the load_mcp factory function."""

    def test_empty_configs_returns_empty_manager(self):
        client = load_mcp([])

        assert isinstance(client, MCPClientManager)
        assert client.get_tool_schemas() == []
        assert client.get_tool_descriptions() == ""

    def test_returns_manager_with_parsed_configs(self):
        configs = [
            {"name": "test", "command": "echo", "args": []},
        ]
        client = load_mcp(configs)

        assert isinstance(client, MCPClientManager)
        assert len(client._configs) == 1
        assert client._configs[0].name == "test"

    def test_no_connection_on_creation(self):
        """load_mcp should be lazy — no connection until start() is called."""
        configs = [
            {"name": "test", "command": "echo", "args": []},
        ]
        client = load_mcp(configs)

        assert client._started is False
        assert len(client._sessions) == 0
        assert len(client._tools) == 0
        assert len(client._tasks) == 0
