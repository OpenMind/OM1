from unittest.mock import AsyncMock, MagicMock, patch

from mcp_servers import load_mcp
from mcp_servers.client import MCPClientManager


class TestLoadMcp:
    """Test the load_mcp factory function."""

    def test_empty_configs_returns_empty_manager(self):
        client = load_mcp([])

        assert client.get_tool_schemas() == []
        assert client.get_tool_descriptions() == ""

    def test_connect_success(self):
        configs = [
            {"name": "test", "transport": "stdio", "command": "echo", "args": []},
        ]

        mock_client = MagicMock()
        mock_client.get_tool_schemas.return_value = [{"type": "function"}]
        mock_client.connect_all = AsyncMock()

        with (
            patch("mcp_servers.MCPClientManager", return_value=mock_client),
            patch("asyncio.get_event_loop") as mock_loop,
        ):
            mock_loop.return_value.is_running.return_value = False

            with patch("asyncio.run", new_callable=MagicMock):
                result = load_mcp(configs)

        assert result == mock_client

    def test_connect_failure_returns_empty_manager(self):
        configs = [
            {"name": "bad", "transport": "stdio", "command": "fail", "args": []},
        ]

        mock_client = MagicMock()
        fallback_client = MCPClientManager([])

        with (
            patch(
                "mcp_servers.MCPClientManager",
                side_effect=[mock_client, fallback_client],
            ),
            patch("asyncio.get_event_loop") as mock_loop,
        ):
            mock_loop.return_value.is_running.return_value = False

            with patch("asyncio.run", side_effect=Exception("connection failed")):
                result = load_mcp(configs)

        assert result.get_tool_schemas() == []
        assert result.is_mcp_tool("anything") is False

    def test_connect_with_running_loop_uses_nest_asyncio(self):
        configs = [
            {"name": "test", "transport": "stdio", "command": "echo", "args": []},
        ]

        mock_client = MagicMock()
        mock_client.get_tool_schemas.return_value = []
        mock_client.connect_all = AsyncMock()

        with (
            patch("mcp_servers.MCPClientManager", return_value=mock_client),
            patch("mcp_servers.nest_asyncio") as mock_nest,
            patch("asyncio.get_event_loop") as mock_loop,
        ):
            loop = MagicMock()
            loop.is_running.return_value = True
            mock_loop.return_value = loop

            result = load_mcp(configs)

            mock_nest.apply.assert_called_once()
            loop.run_until_complete.assert_called_once()
            assert result == mock_client
