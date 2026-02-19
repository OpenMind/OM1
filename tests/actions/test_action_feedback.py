import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from actions.orchestrator import ActionOrchestrator, ActionResult
from llm.output_model import Action
from runtime.config import RuntimeConfig


@dataclass
class FeedbackMockInput:
    action: str


@dataclass
class FeedbackMockOutput:
    result: str


@dataclass
class FeedbackMockInterface(Interface[FeedbackMockInput, FeedbackMockOutput]):
    input: FeedbackMockInput
    output: FeedbackMockOutput


class SuccessConnector(ActionConnector[ActionConfig, FeedbackMockInput]):
    """Connector that always succeeds."""

    async def connect(self, output_interface: FeedbackMockInput) -> None:
        pass

    def tick(self):
        pass


class FailingConnector(ActionConnector[ActionConfig, FeedbackMockInput]):
    """Connector that always raises an exception."""

    async def connect(self, output_interface: FeedbackMockInput) -> None:
        raise TimeoutError("Connection timed out")

    def tick(self):
        pass


def _make_config(agent_actions: list) -> RuntimeConfig:
    config = MagicMock(spec=RuntimeConfig)
    config.action_execution_mode = "concurrent"
    config.action_dependencies = {}
    config.agent_actions = agent_actions
    return config


def _make_agent_action(
    name: str, connector_cls=SuccessConnector
) -> AgentAction:
    connector = connector_cls(ActionConfig())
    return AgentAction(
        name=name,
        llm_label=name,
        interface=FeedbackMockInterface,
        connector=connector,
        exclude_from_prompt=False,
    )


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_success_result(self):
        result = ActionResult(
            action_type="speak", action_value="hello", success=True
        )
        assert result.action_type == "speak"
        assert result.action_value == "hello"
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        result = ActionResult(
            action_type="move",
            action_value="forward",
            success=False,
            error="TimeoutError: Connection timed out",
        )
        assert result.success is False
        assert "TimeoutError" in result.error


class TestFlushPromisesActionResults:
    """Test that flush_promises returns ActionResult objects."""

    @pytest.mark.asyncio
    async def test_successful_action_returns_action_result(self):
        action = _make_agent_action("speak")
        config = _make_config([action])
        orchestrator = ActionOrchestrator(config)

        await orchestrator.promise([Action(type="speak", value="hello")])
        results, pending = await orchestrator.flush_promises()

        assert len(results) == 1
        assert isinstance(results[0], ActionResult)
        assert results[0].action_type == "speak"
        assert results[0].action_value == "hello"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_failed_action_returns_action_result_with_error(self):
        action = _make_agent_action("move", FailingConnector)
        config = _make_config([action])
        orchestrator = ActionOrchestrator(config)

        await orchestrator.promise([Action(type="move", value="forward")])
        results, pending = await orchestrator.flush_promises()

        assert len(results) == 1
        assert results[0].success is False
        assert "TimeoutError" in results[0].error

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        success_action = _make_agent_action("speak")
        fail_action = _make_agent_action("move", FailingConnector)
        config = _make_config([success_action, fail_action])
        orchestrator = ActionOrchestrator(config)

        await orchestrator.promise([
            Action(type="speak", value="hello"),
            Action(type="move", value="forward"),
        ])
        results, pending = await orchestrator.flush_promises()

        assert len(results) == 2
        success_results = [r for r in results if r.success]
        fail_results = [r for r in results if not r.success]
        assert len(success_results) == 1
        assert len(fail_results) == 1

    @pytest.mark.asyncio
    async def test_empty_queue_returns_empty_lists(self):
        config = _make_config([])
        orchestrator = ActionOrchestrator(config)

        results, pending = await orchestrator.flush_promises()
        assert results == []
        assert pending == []

    @pytest.mark.asyncio
    async def test_input_parsing_error_returns_failed_action_result(self):
        """Invalid LLM params should return ActionResult(success=False), not vanish."""
        action = _make_agent_action("speak")
        config = _make_config([action])
        orchestrator = ActionOrchestrator(config)

        # Send JSON with wrong param name - FeedbackMockInput expects "action", not "wrong"
        await orchestrator.promise(
            [Action(type="speak", value='{"wrong_param": "hello"}')]
        )
        results, pending = await orchestrator.flush_promises()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
