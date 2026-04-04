import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.types import TextContent

from mcp_servers.client import (
    MCPClientManager,
    MCPServerConfig,
    MCPTool,
    TransportType,
)


@asynccontextmanager
async def mock_stdio_client(*args, **kwargs):
    yield "read", "write"


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

    def test_server_config(self):
        config = MCPServerConfig(name="test", command="python", args=["-m", "server"])
        assert config.name == "test"
        assert config.transport == TransportType.STDIO

    def test_client_manager_parses_configs(self):
        configs = [
            {"name": "s1", "command": "python", "args": []},
            {"name": "s2", "command": "node", "args": ["-y", "server"]},
        ]
        manager = MCPClientManager(configs)

        assert len(manager._configs) == 2
        assert isinstance(manager._configs[0], MCPServerConfig)
        assert isinstance(manager._configs[1], MCPServerConfig)

    @pytest.mark.asyncio
    async def test_missing_command_raises(self):
        manager = MCPClientManager([{"name": "bad", "transport": "stdio"}])
        event = asyncio.Event()
        await manager._run_server_task(manager._configs[0], event)
        assert event.is_set()
        assert len(manager._sessions) == 0


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
    async def test_stop_clears_state(self):
        manager = self._make_manager_with_tools()
        manager._sessions = {"weather": Mock()}
        manager._started = True
        manager._close_event = asyncio.Event()

        # Add a dummy task that will be cancelled
        async def dummy_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_task())
        manager._tasks = [task]

        await manager.stop()

        assert len(manager._tasks) == 0
        assert len(manager._sessions) == 0
        assert len(manager._tools) == 0
        assert manager._started is False
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self):
        manager = MCPClientManager([])
        await manager.stop()
        assert manager._started is False


class TestConnectAll:
    """Test connect_all with mocked transports."""

    @pytest.mark.asyncio
    async def test_start_discovers_tools(self):
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
                "mcp_servers.client.stdio_client",
                side_effect=mock_stdio_client,
            ),
            patch("mcp_servers.client.ClientSession", return_value=mock_session),
        ):
            await manager.start()

        assert manager._started is True
        assert "mcp_weather_get_weather" in manager._tools
        assert manager._tools["mcp_weather_get_weather"].description == "Get weather info"
        await manager.stop()

    @pytest.mark.asyncio
    async def test_start_handles_server_failure(self):
        configs = [
            {"name": "bad_server", "command": "fail", "args": []},
        ]
        manager = MCPClientManager(configs)

        with patch(
            "mcp_servers.client.stdio_client",
            side_effect=ConnectionError("refused"),
        ):
            await manager.start()

        assert manager._started is True
        assert len(manager._tools) == 0
        assert len(manager._sessions) == 0
        await manager.stop()
