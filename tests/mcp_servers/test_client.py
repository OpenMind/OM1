from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.types import TextContent

from mcp_servers.client import (
    MCPClientManager,
    MCPTool,
    StdioServerConfig,
)


class TestMCPToolSchema:
    """Test MCPTool schema generation."""

    def test_convert_to_schema(self):
        tool = MCPTool(
            key="mcp_weather_get",
            server_name="weather",
            original_name="get",
            description="Get weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        schema = tool.convert_to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp_weather_get"
        assert schema["function"]["description"] == "Get weather"
        assert "city" in schema["function"]["parameters"]["properties"]

    def test_generate_description(self):
        tool = MCPTool(
            key="mcp_weather_get",
            server_name="weather",
            original_name="get",
            description="Get weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        desc = tool.generate_description()

        assert "mcp_weather_get" in desc
        assert "city: string" in desc
        assert "Get weather" in desc


class TestConfigParsing:
    """Test server config validation."""

    def test_stdio_config(self):
        config = StdioServerConfig(name="test", command="python", args=["-m", "server"])
        assert config.name == "test"

    def test_client_manager_parses_configs(self):
        configs = [
            {"name": "s1", "command": "python", "args": []},
            {"name": "s2", "command": "node", "args": ["-y", "server"]},
        ]
        manager = MCPClientManager(configs)

        assert len(manager._configs) == 2
        assert isinstance(manager._configs[0], StdioServerConfig)
        assert isinstance(manager._configs[1], StdioServerConfig)

    def test_missing_command_raises(self):
        with pytest.raises(Exception):
            MCPClientManager([{"name": "bad"}])


class TestMCPClientManager:
    """Test MCPClientManager methods."""

    def _make_manager_with_tools(self):
        """Create a manager with pre-populated tools (no real connection)."""
        manager = MCPClientManager([])
        manager._tools = {
            "mcp_weather_get": MCPTool(
                key="mcp_weather_get",
                server_name="weather",
                original_name="get",
                description="Get weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
            "mcp_slack_post": MCPTool(
                key="mcp_slack_post",
                server_name="slack",
                original_name="post",
                description="Post message",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            ),
        }
        return manager

    def test_get_tool_schemas(self):
        manager = self._make_manager_with_tools()
        schemas = manager.get_tool_schemas()

        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"mcp_weather_get", "mcp_slack_post"}

    def test_get_tool_descriptions_empty(self):
        manager = MCPClientManager([])
        assert manager.get_tool_descriptions() == ""

    def test_get_tool_descriptions_non_empty(self):
        manager = self._make_manager_with_tools()
        desc = manager.get_tool_descriptions()

        assert "mcp_weather_get" in desc
        assert "mcp_slack_post" in desc

    def test_is_mcp_tool(self):
        manager = self._make_manager_with_tools()

        assert manager.is_mcp_tool("mcp_weather_get") is True
        assert manager.is_mcp_tool("mcp_slack_post") is True
        assert manager.is_mcp_tool("speak") is False
        assert manager.is_mcp_tool("unknown") is False

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):

        manager = self._make_manager_with_tools()

        mock_result = Mock()
        mock_result.content = [
            TextContent(type="text", text="sunny 72°F"),
        ]

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        manager._sessions["weather"] = mock_session

        result = await manager.call_tool("mcp_weather_get", {"city": "SF"})

        assert result == "sunny 72°F"
        mock_session.call_tool.assert_called_once_with("get", arguments={"city": "SF"})

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises(self):
        manager = MCPClientManager([])

        with pytest.raises(ValueError, match="Unknown MCP tool"):
            await manager.call_tool("mcp_nonexistent", {})

    @pytest.mark.asyncio
    async def test_close_all_clears_state(self):
        manager = self._make_manager_with_tools()
        manager._sessions = {"weather": Mock()}
        manager._exit_stack = AsyncMock()

        await manager.close_all()

        assert manager._exit_stack is None
        assert len(manager._sessions) == 0
        assert len(manager._tools) == 0

    @pytest.mark.asyncio
    async def test_close_all_handles_error(self):
        manager = self._make_manager_with_tools()
        manager._sessions = {"weather": Mock()}
        mock_stack = AsyncMock()
        mock_stack.aclose = AsyncMock(side_effect=Exception("close error"))
        manager._exit_stack = mock_stack

        await manager.close_all()

        assert manager._exit_stack is None
        assert len(manager._sessions) == 0

    @pytest.mark.asyncio
    async def test_close_all_noop_when_no_stack(self):
        manager = MCPClientManager([])
        manager._exit_stack = None

        await manager.close_all()

        assert manager._exit_stack is None


class TestConnectAll:
    """Test connect_all with mocked transports."""

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self):
        mock_tool = Mock()
        mock_tool.name = "get_weather"
        mock_tool.description = "Get weather info"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[mock_tool]))

        configs = [
            {"name": "weather", "command": "python", "args": []},
        ]
        manager = MCPClientManager(configs)

        with (
            patch(
                "mcp_servers.client.StdioTransport.connect",
                return_value=("read", "write"),
            ),
            patch("mcp_servers.client.ClientSession", return_value=mock_session),
        ):
            await manager.connect_all()

        assert "mcp_weather_get_weather" in manager._tools
        assert (
            manager._tools["mcp_weather_get_weather"].description == "Get weather info"
        )

    @pytest.mark.asyncio
    async def test_connect_handles_server_failure(self):
        configs = [
            {"name": "bad_server", "command": "fail", "args": []},
        ]
        manager = MCPClientManager(configs)

        with patch(
            "mcp_servers.client.StdioTransport.connect",
            side_effect=ConnectionError("refused"),
        ):
            await manager.connect_all()

        assert len(manager._tools) == 0
        assert len(manager._sessions) == 0
