import asyncio
from typing import Any, Dict, List, Optional, Union, cast

import pytest

from llm.output_model import Action, CortexOutputModel
from mcp_servers.orchestrator import MCPOrchestrator, ToolResult
from runtime.config import RuntimeConfig


class MockMCPClient:
    """Mock MCP client that tracks tool calls."""

    def __init__(self, tool_responses: Optional[Dict[str, Union[str, Exception]]] = None):
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


class MockConfig:
    """Thin config wrapper expected by MCPOrchestrator."""

    def __init__(self, mcp_servers, cortex_llm):
        self.mcp_servers = mcp_servers
        self.cortex_llm = cortex_llm


@pytest.fixture
def mock_client():
    return MockMCPClient()


@pytest.fixture
def make_output():
    """Factory for CortexOutputModel."""

    def _make(actions: List[tuple]) -> CortexOutputModel:
        return CortexOutputModel(actions=[Action(type=t, value=v) for t, v in actions])

    return _make


@pytest.fixture
def orch(mock_client):
    llm = MockLLM([])
    return MCPOrchestrator(cast(RuntimeConfig, MockConfig(mock_client, llm)))


class TestInit:
    """MCPOrchestrator.start() connects and injects MCP schemas into the LLM."""

    @pytest.mark.asyncio
    async def test_extends_function_schemas(self, mock_client):
        llm = MockLLM([])
        llm.function_schemas = [{"type": "function", "function": {"name": "speak"}}]

        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(mock_client, llm)))
        await orch.start()

        names = [s["function"]["name"] for s in llm.function_schemas]
        assert "speak" in names
        assert len(names) == 1 + len(mock_client._tools)

    @pytest.mark.asyncio
    async def test_does_not_duplicate_mcp_schemas_on_reinit(self, mock_client):
        llm = MockLLM([])
        llm.function_schemas = [{"type": "function", "function": {"name": "mcp_weather_get"}}]

        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(mock_client, llm)))
        await orch.start()

        names = [s["function"]["name"] for s in llm.function_schemas]
        assert names.count("mcp_weather_get") == 1


class TestActionSplitting:
    """extract_om1_actions filters out MCP tools; execute_mcp_actions handles MCP."""

    @pytest.mark.asyncio
    async def test_extract_mcp_actions_returns_only_mcp(self, orch, make_output):
        output = make_output([("speak", "hi"), ("mcp_weather_get", "{}"), ("emotion", "happy")])
        results, mcp_actions = await orch.execute_mcp_actions(output.actions, set())
        assert results is not None
        assert len(mcp_actions) == 1
        assert mcp_actions[0].type == "mcp_weather_get"

    def test_extract_om1_actions_returns_non_mcp(self, orch, make_output):
        output = make_output([("speak", "hi"), ("mcp_weather_get", "{}"), ("emotion", "happy")])
        om1 = orch.extract_om1_actions(output.actions)
        assert {a.type for a in om1} == {"speak", "emotion"}

    @pytest.mark.asyncio
    async def test_empty_actions_returns_empty_lists(self, orch):
        results, mcp_actions = await orch.execute_mcp_actions([], set())
        assert results is None
        assert mcp_actions is None
        assert orch.extract_om1_actions([]) == []

    def test_all_mcp_returns_empty_om1(self, orch, make_output):
        output = make_output([("mcp_weather_get", "{}"), ("mcp_slack_post", "{}")])
        assert orch.extract_om1_actions(output.actions) == []

    @pytest.mark.asyncio
    async def test_all_om1_returns_empty_mcp(self, orch, make_output):
        output = make_output([("speak", "hi"), ("emotion", "happy")])
        results, mcp_actions = await orch.execute_mcp_actions(output.actions, set())
        assert results is None
        assert mcp_actions is None


class TestExecuteMcpActions:
    """execute_mcp_actions – concurrent tool execution."""

    @pytest.mark.asyncio
    async def test_successful_tool_call(self, make_output):
        client = MockMCPClient(tool_responses={"mcp_weather_get": '{"temp":72}'})
        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(client, MockLLM([]))))

        actions = make_output([("mcp_weather_get", '{"city":"SF"}')]).actions
        results, _ = await orch.execute_mcp_actions(actions, set())

        assert len(results) == 1
        assert results[0].success is True
        assert "72" in results[0].content

    @pytest.mark.asyncio
    async def test_failed_tool_call_marked_failed(self, make_output):
        client = MockMCPClient(tool_responses={"mcp_weather_get": Exception("timeout")})
        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(client, MockLLM([]))))

        actions = make_output([("mcp_weather_get", '{"city":"SF"}')]).actions
        results, _ = await orch.execute_mcp_actions(actions, set())

        assert results[0].success is False
        assert "timeout" in results[0].content

    @pytest.mark.asyncio
    async def test_multiple_tools_executed_concurrently(self, mock_client, make_output):
        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(mock_client, MockLLM([]))))
        actions = make_output([("mcp_weather_get", "{}"), ("mcp_slack_post", "{}")]).actions
        results, _ = await orch.execute_mcp_actions(actions, set())

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_tool_timeout_returns_failed_result(self, make_output):
        async def slow_call(tool_key, args):
            await asyncio.sleep(100)
            return "never"

        client = MockMCPClient()
        client.call_tool = slow_call
        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(client, MockLLM([]))))

        actions = make_output([("mcp_weather_get", '{"city":"SF"}')]).actions
        results, _ = await orch.execute_mcp_actions(actions, set())

        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_dedup_skips_already_succeeded_call(self, make_output):
        client = MockMCPClient(tool_responses={"mcp_weather_get": '{"temp":72}'})
        orch = MCPOrchestrator(cast(RuntimeConfig, MockConfig(client, MockLLM([]))))

        actions = make_output([("mcp_weather_get", '{"city":"SF"}')]).actions
        sig = orch.build_call_signature(actions[0])
        succeeded = {sig}

        results, mcp_actions = await orch.execute_mcp_actions(actions, succeeded)
        assert results is None
        assert mcp_actions is None


class TestBuildResultPrompt:
    """build_result_prompt formats the recall prompt correctly."""

    def test_includes_original_prompt(self, orch):
        results = [ToolResult("mcp_weather_get", True, '{"temp":72}')]
        prompt = orch.build_result_prompt("What is the weather?", results)
        assert "What is the weather?" in prompt

    def test_includes_tool_results_block(self, orch):
        results = [ToolResult("mcp_weather_get", True, '{"temp":72}')]
        prompt = orch.build_result_prompt("prompt", results)
        assert "[Tool Results]" in prompt
        assert "mcp_weather_get" in prompt

    def test_failed_result_marked_failed(self, orch):
        results = [ToolResult("mcp_weather_get", False, "Error: timeout")]
        prompt = orch.build_result_prompt("prompt", results)
        assert "FAILED" in prompt

    def test_successful_result_marked_ok(self, orch):
        results = [ToolResult("mcp_weather_get", True, "sunny")]
        prompt = orch.build_result_prompt("prompt", results)
        assert "OK" in prompt

    def test_multiple_results_all_included(self, orch):
        results = [
            ToolResult("mcp_weather_get", True, "sunny"),
            ToolResult("mcp_slack_post", False, "Error"),
        ]
        prompt = orch.build_result_prompt("prompt", results)
        assert "mcp_weather_get" in prompt
        assert "mcp_slack_post" in prompt


class TestBuildCallSignature:
    """build_call_signature – deterministic dedup fingerprint."""

    def test_same_action_produces_same_signature(self, orch):
        a1 = Action(type="mcp_weather_get", value='{"city":"SF"}')
        a2 = Action(type="mcp_weather_get", value='{"city":"SF"}')
        assert orch.build_call_signature(a1) == orch.build_call_signature(a2)

    def test_different_args_produce_different_signatures(self, orch):
        a1 = Action(type="mcp_weather_get", value='{"city":"SF"}')
        a2 = Action(type="mcp_weather_get", value='{"city":"NY"}')
        assert orch.build_call_signature(a1) != orch.build_call_signature(a2)

    def test_different_tools_produce_different_signatures(self, orch):
        a1 = Action(type="mcp_weather_get", value="{}")
        a2 = Action(type="mcp_slack_post", value="{}")
        assert orch.build_call_signature(a1) != orch.build_call_signature(a2)

    def test_key_order_does_not_affect_signature(self, orch):
        a1 = Action(type="mcp_weather_get", value='{"city":"SF","units":"metric"}')
        a2 = Action(type="mcp_weather_get", value='{"units":"metric","city":"SF"}')
        assert orch.build_call_signature(a1) == orch.build_call_signature(a2)
