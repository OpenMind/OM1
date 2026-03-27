import asyncio
from typing import Any, Dict, List, Optional, Union

import pytest

from llm.output_model import Action, CortexOutputModel
from mcp_servers.orchestrator import MCPOrchestrator, ToolResult


class MockMCPClient:
    """Mock MCP client that tracks tool calls."""

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

    async def close_all(self):
        pass


class MockLLM:
    """Mock LLM that returns predefined outputs per call."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self._call_count = 0
        self.function_schemas: list = []
        self._skip_state_management = False

    async def ask(self, prompt: str) -> Any:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return None


@pytest.fixture
def mock_client():
    return MockMCPClient()


@pytest.fixture
def make_output():
    """Factory for CortexOutputModel."""

    def _make(actions: List[tuple]) -> CortexOutputModel:
        return CortexOutputModel(actions=[Action(type=t, value=v) for t, v in actions])

    return _make


class TestInit:
    """Test MCPOrchestrator initialization."""

    def test_extends_function_schemas(self, mock_client):
        llm = MockLLM([])
        llm.function_schemas = [{"type": "function", "function": {"name": "speak"}}]

        MCPOrchestrator(mock_client, llm)

        names = [s["function"]["name"] for s in llm.function_schemas]
        assert "speak" in names
        assert len(names) == 1 + len(mock_client._tools)


class TestProcessNoMCP:
    """Test process() when there are no MCP actions."""

    @pytest.mark.asyncio
    async def test_no_actions_returns_output(self, mock_client, make_output):
        llm = MockLLM([])
        orch = MCPOrchestrator(mock_client, llm)

        output = make_output([("speak", "hello"), ("emotion", "happy")])
        result = await orch.process(output, "test prompt", llm)

        assert result is not None
        assert len(result.actions) == 2
        assert mock_client.calls == []

    @pytest.mark.asyncio
    async def test_none_input_returns_none(self, mock_client):
        llm = MockLLM([])
        orch = MCPOrchestrator(mock_client, llm)

        result = await orch.process(None, "test prompt", llm)
        assert result is None


class TestProcessWithMCP:
    """Test process() with MCP tool calls."""

    @pytest.mark.asyncio
    async def test_single_round_mcp(self, mock_client, make_output):
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        final = make_output([("speak", "73°F"), ("emotion", "happy")])

        llm = MockLLM([final])
        orch = MCPOrchestrator(mock_client, llm)

        result = await orch.process(initial, "weather?", llm)

        assert len(mock_client.calls) == 1
        assert mock_client.calls[0][0] == "mcp_weather_get"
        assert len(result.actions) == 2
        assert all(not a.type.startswith("mcp_") for a in result.actions)

    @pytest.mark.asyncio
    async def test_multi_round_mcp(self, mock_client, make_output):
        initial = make_output([("mcp_maps_geocode", '{"address":"SF"}')])
        round1 = make_output([("mcp_weather_get", '{"lat":37}')])
        final = make_output([("speak", "sunny")])

        llm = MockLLM([round1, final])
        orch = MCPOrchestrator(mock_client, llm)

        result = await orch.process(initial, "weather?", llm)

        assert len(mock_client.calls) == 2
        assert mock_client.calls[0][0] == "mcp_maps_geocode"
        assert mock_client.calls[1][0] == "mcp_weather_get"
        assert result.actions[0].type == "speak"

    @pytest.mark.asyncio
    async def test_strips_mcp_from_final(self, mock_client, make_output):
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        # LLM returns mix of MCP + OM1 actions, then pure OM1
        round1 = make_output(
            [
                ("speak", "sunny"),
                ("mcp_slack_post", '{"text":"hi"}'),
                ("emotion", "happy"),
            ]
        )
        final = make_output([("speak", "done"), ("emotion", "happy")])

        llm = MockLLM([round1, final])
        orch = MCPOrchestrator(mock_client, llm)

        result = await orch.process(initial, "test", llm)

        types = [a.type for a in result.actions]
        assert "speak" in types
        assert "emotion" in types
        assert not any(t.startswith("mcp_") for t in types)

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self, mock_client, make_output):
        """Process stops after max_rounds even if LLM keeps requesting MCP."""
        mcp_output = make_output([("mcp_weather_get", '{"city":"SF"}')])
        llm = MockLLM([mcp_output, mcp_output, mcp_output, mcp_output, mcp_output])
        orch = MCPOrchestrator(mock_client, llm)

        await orch.process(mcp_output, "test", llm, max_rounds=3)

        assert len(mock_client.calls) <= 3


class TestDeduplication:
    """Test that identical tool+args calls are skipped."""

    @pytest.mark.asyncio
    async def test_skips_duplicate_calls(self, mock_client, make_output):
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        # LLM requests the exact same tool again
        duplicate = make_output([("mcp_weather_get", '{"city":"SF"}')])
        final = make_output([("speak", "sunny")])

        llm = MockLLM([duplicate, final])
        orch = MCPOrchestrator(mock_client, llm)

        await orch.process(initial, "test", llm)

        # Only called once, second was deduped
        assert len(mock_client.calls) == 1

    @pytest.mark.asyncio
    async def test_different_args_not_deduped(self, mock_client, make_output):
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        different = make_output([("mcp_weather_get", '{"city":"LA"}')])
        final = make_output([("speak", "done")])

        llm = MockLLM([different, final])
        orch = MCPOrchestrator(mock_client, llm)

        await orch.process(initial, "test", llm)

        assert len(mock_client.calls) == 2


class TestErrorHandling:
    """Test tool execution failure handling."""

    @pytest.mark.asyncio
    async def test_tool_exception_marked_failed(self, make_output):
        client = MockMCPClient(
            tool_responses={
                "mcp_weather_get": Exception("connection refused"),
            }
        )
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        final = make_output([("speak", "sorry")])

        llm = MockLLM([final])
        orch = MCPOrchestrator(client, llm)  # type: ignore[arg-type]

        result = await orch.process(initial, "test", llm)

        assert result.actions[0].value == "sorry"

    @pytest.mark.asyncio
    async def test_tool_timeout(self, make_output):
        """Test that tool timeout is handled gracefully."""

        async def slow_call(tool_key, args):
            await asyncio.sleep(100)
            return "never"

        client = MockMCPClient()
        client.call_tool = slow_call
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        final = make_output([("speak", "timed out")])

        llm = MockLLM([final])
        orch = MCPOrchestrator(client, llm)  # type: ignore[arg-type]

        result = await orch.process(initial, "test", llm)

        assert result is not None

    @pytest.mark.asyncio
    async def test_llm_returns_none(self, mock_client, make_output):
        """LLM timeout returns None → process returns None."""
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])

        llm = MockLLM([None])
        orch = MCPOrchestrator(mock_client, llm)

        result = await orch.process(initial, "test", llm)

        assert result is None


class TestDispatchOM1:
    """Test OM1 action dispatching during MCP rounds."""

    @pytest.mark.asyncio
    async def test_dispatches_om1_actions(self, mock_client, make_output):
        initial = make_output(
            [
                ("emotion", "think"),
                ("mcp_weather_get", '{"city":"SF"}'),
            ]
        )
        final = make_output([("speak", "done")])

        dispatched = []

        async def mock_dispatch(actions):
            dispatched.extend(actions)

        llm = MockLLM([final])
        orch = MCPOrchestrator(mock_client, llm)

        await orch.process(initial, "test", llm, dispatch_om1=mock_dispatch)

        assert len(dispatched) == 1
        assert dispatched[0].type == "emotion"


class TestHistoryManagement:
    """Test that recall_llm allows normal history management."""

    @pytest.mark.asyncio
    async def test_flag_not_set_during_recall(self, mock_client, make_output):
        """MCP recall should NOT skip state management."""
        initial = make_output([("mcp_weather_get", '{"city":"SF"}')])
        final = make_output([("speak", "done")])

        flags_during_ask = []

        class TrackingLLM(MockLLM):
            async def ask(self, prompt):
                flags_during_ask.append(self._skip_state_management)
                return await super().ask(prompt)

        llm = TrackingLLM([final])
        orch = MCPOrchestrator(mock_client, llm)

        await orch.process(initial, "test", llm)

        assert flags_during_ask[-1] is False


class TestBuildResultPrompt:
    """Test _build_result_prompt output format."""

    def test_includes_tool_results(self, mock_client):
        llm = MockLLM([])
        orch = MCPOrchestrator(mock_client, llm)

        results = [ToolResult("mcp_weather_get", True, '{"temp":73}')]
        prompt = orch._build_result_prompt("original", results)

        assert "original" in prompt
        assert "mcp_weather_get" in prompt
        assert '{"temp":73}' in prompt
        assert "OK" in prompt

    def test_marks_failed_tools(self, mock_client):
        llm = MockLLM([])
        orch = MCPOrchestrator(mock_client, llm)

        results = [ToolResult("mcp_slack_post", False, "Error: timeout")]
        prompt = orch._build_result_prompt("original", results)

        assert "FAILED" in prompt
        assert "Error: timeout" in prompt

    def test_mixed_results(self, mock_client):
        llm = MockLLM([])
        orch = MCPOrchestrator(mock_client, llm)

        results = [
            ToolResult("mcp_weather_get", True, "ok"),
            ToolResult("mcp_maps_geocode", False, "error"),
        ]
        prompt = orch._build_result_prompt("original", results)

        assert "[mcp_weather_get] OK" in prompt
        assert "[mcp_maps_geocode] FAILED" in prompt


class TestParseArguments:
    """Test _parse_arguments with various input formats."""

    @pytest.fixture
    def orch(self, mock_client):
        llm = MockLLM([])
        return MCPOrchestrator(mock_client, llm)

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
