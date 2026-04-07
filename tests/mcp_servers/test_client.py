from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import TextContent

from mcp_servers.client import (
    MCPClientManager,
    MCPServerConfig,
    MCPTool,
    TransportType,
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
        config = MCPServerConfig(
            name="test",
            transport=TransportType.STDIO,
            command="python",
            args=["-m", "server"],
        )
        assert config.name == "test"
        assert config.transport == TransportType.STDIO

    def test_sse_config(self):
        config = MCPServerConfig(
            name="remote", transport=TransportType.SSE, url="http://localhost:8080/sse"
        )
        assert config.transport == TransportType.SSE
        assert config.url == "http://localhost:8080/sse"

    def test_http_config(self):
        config = MCPServerConfig(
            name="remote", transport=TransportType.HTTP, url="http://localhost:8080/mcp"
        )
        assert config.transport == TransportType.HTTP

    def test_client_manager_parses_stdio_configs(self):
        configs = [
            {"name": "s1", "transport": "stdio", "command": "python", "args": []},
            {
                "name": "s2",
                "transport": "stdio",
                "command": "node",
                "args": ["-y", "server"],
            },
        ]
        manager = MCPClientManager(configs)

        assert len(manager._configs) == 2
        assert isinstance(manager._configs[0], MCPServerConfig)
        assert manager._configs[0].transport == TransportType.STDIO

    def test_client_manager_default_transport_stdio(self):
        """If transport not specified, defaults to stdio."""
        configs = [{"name": "s1", "command": "python", "args": []}]
        manager = MCPClientManager(configs)
        assert manager._configs[0].transport == TransportType.STDIO


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
        manager._started = True
        close_event = AsyncMock()
        close_event.set = Mock()
        manager._close_event = close_event

        # No real tasks — just verify stop() doesn't error and resets state
        manager._tasks = []
        await manager.stop()

        assert manager._started is False
        assert len(manager._sessions) == 0
        assert len(manager._tools) == 0

    @pytest.mark.asyncio
    async def test_start_noop_when_already_started(self):
        manager = MCPClientManager([])
        manager._started = True

        # Should be a no-op
        await manager.start()
        assert manager._started is True


class TestTransportConnectDispatch:
    """Test that _connect_server dispatches to the right transport."""

    def test_stdio_transport_dispatches(self):
        """Verify stdio config is parsed and stored correctly."""
        configs = [
            {"name": "weather", "transport": "stdio", "command": "python", "args": []}
        ]
        mgr = MCPClientManager(configs)

        assert len(mgr._configs) == 1
        assert mgr._configs[0].transport == TransportType.STDIO
        assert mgr._configs[0].command == "python"
        assert mgr._started is False

    @pytest.mark.asyncio
    async def test_sse_missing_url_raises(self):
        """SSE config without url should raise on connect."""
        configs = [{"name": "remote", "transport": "sse"}]
        manager = MCPClientManager(configs)

        from contextlib import AsyncExitStack

        exit_stack = AsyncExitStack()
        async with exit_stack:
            with pytest.raises(ValueError, match="requires 'url'"):
                await manager._connect_server(manager._configs[0], exit_stack)

    @pytest.mark.asyncio
    async def test_http_missing_url_raises(self):
        """HTTP config without url should raise on connect."""
        configs = [{"name": "remote", "transport": "http"}]
        manager = MCPClientManager(configs)

        from contextlib import AsyncExitStack

        exit_stack = AsyncExitStack()
        async with exit_stack:
            with pytest.raises(ValueError, match="requires 'url'"):
                await manager._connect_server(manager._configs[0], exit_stack)

    @pytest.mark.asyncio
    async def test_stdio_missing_command_raises(self):
        """Stdio config without command should raise on connect."""
        configs = [{"name": "local", "transport": "stdio"}]
        manager = MCPClientManager(configs)

        from contextlib import AsyncExitStack

        exit_stack = AsyncExitStack()
        async with exit_stack:
            with pytest.raises(ValueError, match="requires 'command'"):
                await manager._connect_server(manager._configs[0], exit_stack)
