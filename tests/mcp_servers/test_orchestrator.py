import asyncio
from typing import Dict, List, Optional, Union
from unittest.mock import patch

import pytest

from llm.output_model import Action, CortexOutputModel
from mcp_servers.orchestrator import MCPOrchestrator, ToolResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class MockMCPClient:
    """Fake MCPClientManager that tracks call_tool invocations."""

    def __init__(
        self, tool_responses: Optional[Dict[str, Union[str, Exception]]] = None
    ):
        self._tools = {"mcp_weather_get", "mcp_slack_post", "mcp_maps_geocode"}
        self._responses = tool_responses or {}
        self.calls: List[tuple] = []

    def get_tool_schemas(self) -> list:
        return [
            {
                "type": "function",
                "function": {"name": name, "parameters": {}},
            }
            for name in self._tools
        ]

    def is_mcp_tool(self, tool_type: str) -> bool:
        return tool_type in self._tools

    async def call_tool(self, tool_key: str, args: dict) -> str:
        self.calls.append((tool_key, args))
        if tool_key in self._responses:
            resp = self._responses[tool_key]
            if isinstance(resp, Exception):
                raise resp
            return resp
        return f'{{"ok":true,"tool":"{tool_key}"}}'

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class MockLLM:
    """Mock LLM with a populated function_schemas list."""

    def __init__(self):
        self.function_schemas: list = []


class MockConfig:
    """Minimal RuntimeConfig stand-in for MCPOrchestrator tests."""

    def __init__(
        self,
        mcp_client: MockMCPClient,
        llm: Optional[MockLLM] = None,
    ):
        self.mcp_servers = mcp_client
        self.cortex_llm = llm or MockLLM()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_output(actions: List[tuple]) -> CortexOutputModel:
    return CortexOutputModel(actions=[Action(type=t, value=v) for t, v in actions])


@pytest.fixture
def mock_client():
    return MockMCPClient()


@pytest.fixture
def orch(mock_client):
    """MCPOrchestrator wired to a MockMCPClient via MockConfig."""
    config = MockConfig(mock_client)
    return MCPOrchestrator(config)


# ---------------------------------------------------------------------------
# Init / start
# ---------------------------------------------------------------------------


class TestInit:
    """Test MCPOrchestrator initialization."""

    def test_extends_function_schemas_on_start(self, mock_client):
        llm = MockLLM()
        llm.function_schemas = [{"type": "function", "function": {"name": "speak"}}]
        config = MockConfig(mock_client, llm)
        MCPOrchestrator(config)

        # Before start(), schemas not yet injected
        assert len(llm.function_schemas) == 1

    @pytest.mark.asyncio
    async def test_start_injects_mcp_schemas(self, mock_client):
        llm = MockLLM()
        llm.function_schemas = [{"type": "function", "function": {"name": "speak"}}]
        config = MockConfig(mock_client, llm)
        orc = MCPOrchestrator(config)

        with patch.object(mock_client, "start", return_value=None):
            await orc.start()

        names = {s["function"]["name"] for s in llm.function_schemas}
        assert "speak" in names
        assert len(names) == 1 + len(mock_client._tools)


# ---------------------------------------------------------------------------
# execute_mcp_actions
# ---------------------------------------------------------------------------


class TestExecuteMCPActions:
    """Test the execute_mcp_actions() public API."""

    @pytest.mark.asyncio
    async def test_no_mcp_actions_returns_none(self, orch):
        output = make_output([("speak", "hello"), ("emotion", "happy")])
        results, mcp_acts = await orch.execute_mcp_actions(output.actions, set())
        assert results is None
        assert mcp_acts is None

    @pytest.mark.asyncio
    async def test_executes_mcp_tool(self, orch, mock_client):
        output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        results, mcp_acts = await orch.execute_mcp_actions(output.actions, set())

        assert results is not None
        assert len(results) == 1
        assert results[0].success is True
        assert len(mock_client.calls) == 1
        assert mock_client.calls[0][0] == "mcp_weather_get"

    @pytest.mark.asyncio
    async def test_deduplicates_succeeded_calls(self, orch, mock_client):
        output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        succeeded: set = set()

        await orch.execute_mcp_actions(output.actions, succeeded)
        assert len(mock_client.calls) == 1

        # Second call with the same set — already succeeded, should skip
        results, _ = await orch.execute_mcp_actions(output.actions, succeeded)
        assert results is None
        assert len(mock_client.calls) == 1  # not called again

    @pytest.mark.asyncio
    async def test_different_args_not_deduped(self, orch, mock_client):
        succeeded: set = set()

        sf_output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        la_output = make_output([("mcp_weather_get", '{"city":"LA"}')])

        await orch.execute_mcp_actions(sf_output.actions, succeeded)
        await orch.execute_mcp_actions(la_output.actions, succeeded)

        assert len(mock_client.calls) == 2

    @pytest.mark.asyncio
    async def test_tool_exception_marked_failed(self):
        client = MockMCPClient(
            tool_responses={"mcp_weather_get": Exception("connection refused")}
        )
        config = MockConfig(client)
        orc = MCPOrchestrator(config)

        output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        results, _ = await orc.execute_mcp_actions(output.actions, set())

        assert results is not None
        assert results[0].success is False
        assert "connection refused" in results[0].content

    @pytest.mark.asyncio
    async def test_tool_timeout_marked_failed(self):
        async def slow_call(tool_key, args):
            await asyncio.sleep(100)
            return "never"

        client = MockMCPClient()
        client.call_tool = slow_call
        config = MockConfig(client)
        orc = MCPOrchestrator(config, max_concurrency=1)

        output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        # Should return within timeout and mark as failed
        results, _ = await orc.execute_mcp_actions(output.actions, set())
        assert results is not None
        assert results[0].success is False


# ---------------------------------------------------------------------------
# extract_om1_actions
# ---------------------------------------------------------------------------


class TestExtractOM1Actions:
    def test_filters_out_mcp_tools(self, orch):
        actions = [
            Action(type="emotion", value="happy"),
            Action(type="mcp_weather_get", value="{}"),
            Action(type="speak", value="done"),
        ]
        om1 = orch.extract_om1_actions(actions)
        types = [a.type for a in om1]
        assert "emotion" in types
        assert "speak" in types
        assert "mcp_weather_get" not in types

    def test_empty_list(self, orch):
        assert orch.extract_om1_actions([]) == []

    def test_all_mcp_returns_empty(self, orch):
        actions = [
            Action(type="mcp_weather_get", value="{}"),
            Action(type="mcp_slack_post", value="{}"),
        ]
        assert orch.extract_om1_actions(actions) == []


# ---------------------------------------------------------------------------
# build_result_prompt
# ---------------------------------------------------------------------------


class TestBuildResultPrompt:
    """Test build_result_prompt() output format."""

    def test_includes_tool_results(self, orch):
        results = [ToolResult("mcp_weather_get", True, '{"temp":73}')]
        prompt = orch.build_result_prompt("original", results)

        assert "original" in prompt
        assert "mcp_weather_get" in prompt
        assert '{"temp":73}' in prompt
        assert "OK" in prompt

    def test_marks_failed_tools(self, orch):
        results = [ToolResult("mcp_slack_post", False, "Error: timeout")]
        prompt = orch.build_result_prompt("original", results)

        assert "FAILED" in prompt
        assert "Error: timeout" in prompt

    def test_mixed_results(self, orch):
        results = [
            ToolResult("mcp_weather_get", True, "ok"),
            ToolResult("mcp_maps_geocode", False, "error"),
        ]
        prompt = orch.build_result_prompt("original", results)

        assert "[mcp_weather_get] OK" in prompt
        assert "[mcp_maps_geocode] FAILED" in prompt


# ---------------------------------------------------------------------------
# _parse_arguments
# ---------------------------------------------------------------------------


class TestParseArguments:
    """Test _parse_arguments with various input formats."""

    def test_json_string(self, orch):
        action = Action(type="test", value='{"city": "SF", "units": "fahrenheit"}')
        result = orch._parse_arguments(action)
        assert result == {"city": "SF", "units": "fahrenheit"}

    def test_plain_string(self, orch):
        action = Action(type="test", value="hello world")
        result = orch._parse_arguments(action)
        assert result == {"action": "hello world"}

    def test_json_array_fallback(self, orch):
        action = Action(type="test", value='["a", "b"]')
        result = orch._parse_arguments(action)
        assert result == {"action": '["a", "b"]'}

    def test_empty_json(self, orch):
        action = Action(type="test", value="{}")
        result = orch._parse_arguments(action)
        assert result == {}
