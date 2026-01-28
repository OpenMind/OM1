import asyncio
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from actions.orchestrator import ActionOrchestrator
from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig


@dataclass
class MockInput:
    action: str


@dataclass
class MockOutput:
    result: str


@dataclass
class MockInterface(Interface[MockInput, MockOutput]):
    input: MockInput
    output: MockOutput


class MockConnector(ActionConnector[ActionConfig, MockInput]):
    """
    Mock connector that tracks execution order and timing.
    """

    execution_order: List[str] = []
    execution_times: Dict[str, float] = {}

    def __init__(self, config: ActionConfig, action_name: str):
        super().__init__(config)
        self.action_name = action_name
        self.connected_values: List[str] = []
        self.tick_count = 0

    async def connect(self, output_interface: MockInput) -> None:
        """Record when this action executes."""
        MockConnector.execution_order.append(self.action_name)
        MockConnector.execution_times[self.action_name] = (
            asyncio.get_event_loop().time()
        )

        self.connected_values.append(output_interface.action)

        await asyncio.sleep(0.01)

    def tick(self):
        """Background connector tick."""
        self.tick_count += 1

    @classmethod
    def reset(cls):
        """Reset class-level tracking."""
        cls.execution_order = []
        cls.execution_times = {}


@pytest.fixture
def mock_runtime_config():
    """Create a mock RuntimeConfig for testing."""
    config = MagicMock(spec=RuntimeConfig)
    config.action_execution_mode = "concurrent"
    config.action_dependencies = {}
    config.agent_actions = []
    return config


@pytest.fixture
def create_agent_action():
    """Factory to create test agent actions."""

    def _create(name: str, llm_label: str) -> AgentAction:
        connector = MockConnector(ActionConfig(), name)
        return AgentAction(
            name=name,
            llm_label=llm_label,
            interface=MockInterface,
            connector=connector,
            exclude_from_prompt=False,
        )

    return _create


class TestActionOrchestratorConcurrent:
    """Test concurrent execution mode (default)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset mock connector state before each test."""
        MockConnector.reset()

    @pytest.mark.asyncio
    async def test_concurrent_execution_all_start_together(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that concurrent mode starts all actions at the same time."""
        action1 = create_agent_action("move", "move")
        action2 = create_agent_action("speak", "speak")
        action3 = create_agent_action("gesture", "gesture")

        mock_runtime_config.agent_actions = [action1, action2, action3]
        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="move", value="forward"),
            Action(type="speak", value="hello"),
            Action(type="gesture", value="wave"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        assert len(MockConnector.execution_order) == 3
        assert set(MockConnector.execution_order) == {"move", "speak", "gesture"}

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that concurrent actions start almost simultaneously."""
        action1 = create_agent_action("action1", "action1")
        action2 = create_agent_action("action2", "action2")

        mock_runtime_config.agent_actions = [action1, action2]
        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action1", value="test1"),
            Action(type="action2", value="test2"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        times = list(MockConnector.execution_times.values())
        assert len(times) == 2
        time_diff = abs(times[1] - times[0])
        assert time_diff < 0.001


class TestActionOrchestratorSequential:
    """Test sequential execution mode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset mock connector state before each test."""
        MockConnector.reset()

    @pytest.mark.asyncio
    async def test_sequential_execution_order(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that sequential mode executes actions in order."""
        action1 = create_agent_action("first", "first")
        action2 = create_agent_action("second", "second")
        action3 = create_agent_action("third", "third")

        mock_runtime_config.agent_actions = [action1, action2, action3]
        mock_runtime_config.action_execution_mode = "sequential"

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="first", value="1"),
            Action(type="second", value="2"),
            Action(type="third", value="3"),
        ]

        await orchestrator.promise(actions)

        assert MockConnector.execution_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_sequential_execution_timing(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that sequential actions execute one after another."""
        action1 = create_agent_action("action1", "action1")
        action2 = create_agent_action("action2", "action2")

        mock_runtime_config.agent_actions = [action1, action2]
        mock_runtime_config.action_execution_mode = "sequential"

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action1", value="test1"),
            Action(type="action2", value="test2"),
        ]

        await orchestrator.promise(actions)

        time1 = MockConnector.execution_times["action1"]
        time2 = MockConnector.execution_times["action2"]
        assert time2 > time1 + 0.009

    @pytest.mark.asyncio
    async def test_sequential_with_single_action(
        self, mock_runtime_config, create_agent_action
    ):
        """Test sequential mode with just one action."""
        action = create_agent_action("solo", "solo")

        mock_runtime_config.agent_actions = [action]
        mock_runtime_config.action_execution_mode = "sequential"

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [Action(type="solo", value="alone")]
        await orchestrator.promise(actions)

        assert MockConnector.execution_order == ["solo"]


class TestActionOrchestratorDependencies:
    """Test dependency-based execution mode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset mock connector state before each test."""
        MockConnector.reset()

    @pytest.mark.asyncio
    async def test_simple_dependency_chain(
        self, mock_runtime_config, create_agent_action
    ):
        """Test simple dependency: B depends on A."""
        action_a = create_agent_action("action_a", "action_a")
        action_b = create_agent_action("action_b", "action_b")

        mock_runtime_config.agent_actions = [action_a, action_b]
        mock_runtime_config.action_execution_mode = "dependencies"
        mock_runtime_config.action_dependencies = {"action_b": ["action_a"]}

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action_b", value="second"),
            Action(type="action_a", value="first"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        assert MockConnector.execution_order.index(
            "action_a"
        ) < MockConnector.execution_order.index("action_b")

    @pytest.mark.asyncio
    async def test_multiple_dependencies(
        self, mock_runtime_config, create_agent_action
    ):
        """Test action waiting for multiple dependencies."""
        action_a = create_agent_action("action_a", "action_a")
        action_b = create_agent_action("action_b", "action_b")
        action_c = create_agent_action("action_c", "action_c")

        mock_runtime_config.agent_actions = [action_a, action_b, action_c]
        mock_runtime_config.action_execution_mode = "dependencies"
        mock_runtime_config.action_dependencies = {"action_c": ["action_a", "action_b"]}

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action_c", value="last"),
            Action(type="action_b", value="middle"),
            Action(type="action_a", value="first"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        c_index = MockConnector.execution_order.index("action_c")
        a_index = MockConnector.execution_order.index("action_a")
        b_index = MockConnector.execution_order.index("action_b")

        assert c_index > a_index
        assert c_index > b_index

    @pytest.mark.asyncio
    async def test_parallel_with_dependencies(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that independent actions can run in parallel while respecting dependencies."""
        action_a = create_agent_action("action_a", "action_a")
        action_b = create_agent_action("action_b", "action_b")
        action_c = create_agent_action("action_c", "action_c")

        mock_runtime_config.agent_actions = [action_a, action_b, action_c]
        mock_runtime_config.action_execution_mode = "dependencies"
        mock_runtime_config.action_dependencies = {"action_c": ["action_b"]}

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action_a", value="independent"),
            Action(type="action_b", value="prerequisite"),
            Action(type="action_c", value="dependent"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        time_a = MockConnector.execution_times["action_a"]
        time_b = MockConnector.execution_times["action_b"]
        time_c = MockConnector.execution_times["action_c"]

        assert abs(time_a - time_b) < 0.001
        assert time_c > time_b + 0.009

    @pytest.mark.asyncio
    async def test_complex_dependency_graph(
        self, mock_runtime_config, create_agent_action
    ):
        """Test complex dependency graph: finale depends on gesture, gesture depends on speak."""
        speak = create_agent_action("speak", "speak")
        gesture = create_agent_action("gesture", "gesture")
        move = create_agent_action("move", "move")
        finale = create_agent_action("finale", "finale")

        mock_runtime_config.agent_actions = [speak, gesture, move, finale]
        mock_runtime_config.action_execution_mode = "dependencies"
        mock_runtime_config.action_dependencies = {
            "gesture": ["speak"],
            "finale": ["gesture", "move"],
        }

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="finale", value="end"),
            Action(type="gesture", value="wave"),
            Action(type="move", value="forward"),
            Action(type="speak", value="hello"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        order = MockConnector.execution_order
        assert order.index("speak") < order.index("gesture")
        assert order.index("gesture") < order.index("finale")
        assert order.index("move") < order.index("finale")

    @pytest.mark.asyncio
    async def test_no_dependencies_acts_like_concurrent(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that dependency mode with no dependencies acts like concurrent mode."""
        action1 = create_agent_action("action1", "action1")
        action2 = create_agent_action("action2", "action2")

        mock_runtime_config.agent_actions = [action1, action2]
        mock_runtime_config.action_execution_mode = "dependencies"
        mock_runtime_config.action_dependencies = {}  # No dependencies

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="action1", value="test1"),
            Action(type="action2", value="test2"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        times = list(MockConnector.execution_times.values())
        time_diff = abs(times[1] - times[0])
        assert time_diff < 0.001


class TestActionOrchestratorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset mock connector state before each test."""
        MockConnector.reset()

    @pytest.mark.asyncio
    async def test_nonexistent_action(self, mock_runtime_config, create_agent_action):
        """Test that nonexistent actions are handled gracefully."""
        action = create_agent_action("real_action", "real_action")

        mock_runtime_config.agent_actions = [action]
        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="real_action", value="exists"),
            Action(type="fake_action", value="does not exist"),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        assert MockConnector.execution_order == ["real_action"]

    @pytest.mark.asyncio
    async def test_action_normalization(self, mock_runtime_config, create_agent_action):
        """Test that action shortcuts are normalized correctly."""
        move_action = create_agent_action("move", "move")

        mock_runtime_config.agent_actions = [move_action]
        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [
            Action(type="stand still", value=""),
            Action(type="turn left", value=""),
            Action(type="move forwards", value=""),
        ]

        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        connector = move_action.connector
        assert "stand still" in connector.connected_values
        assert "turn left" in connector.connected_values
        assert "move forwards" in connector.connected_values

    @pytest.mark.asyncio
    async def test_empty_action_list(self, mock_runtime_config):
        """Test handling of empty action list."""
        orchestrator = ActionOrchestrator(mock_runtime_config)

        await orchestrator.promise([])
        done, pending = await orchestrator.flush_promises()

        assert len(done) == 0
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_promise_queue_tracking(
        self, mock_runtime_config, create_agent_action
    ):
        """Test that promise queue correctly tracks pending and completed actions."""
        action = create_agent_action("test", "test")
        mock_runtime_config.agent_actions = [action]

        orchestrator = ActionOrchestrator(mock_runtime_config)

        actions = [Action(type="test", value=f"action{i}") for i in range(3)]
        await orchestrator.promise(actions)

        assert len(orchestrator.promise_queue) == 3

        done, pending = await orchestrator.flush_promises()
        assert len(done) == 3
        assert len(pending) == 0

    def test_orchestrator_stop(self, mock_runtime_config):
        """Test that orchestrator stops cleanly."""
        orchestrator = ActionOrchestrator(mock_runtime_config)
        orchestrator.start()
        orchestrator.stop()

        assert orchestrator._stop_event.is_set()


class TestActionOrchestratorModeComparison:
    """Compare behavior across different execution modes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset mock connector state before each test."""
        MockConnector.reset()

    @pytest.mark.asyncio
    async def test_same_result_different_order(self, create_agent_action):
        """Test that all modes produce the same result, just in different order."""
        config_concurrent = MagicMock(spec=RuntimeConfig)
        action1 = create_agent_action("action1", "action1")
        action2 = create_agent_action("action2", "action2")
        config_concurrent.agent_actions = [action1, action2]
        config_concurrent.action_execution_mode = "concurrent"
        config_concurrent.action_dependencies = {}

        orchestrator = ActionOrchestrator(config_concurrent)
        actions = [
            Action(type="action1", value="test1"),
            Action(type="action2", value="test2"),
        ]
        await orchestrator.promise(actions)
        await orchestrator.flush_promises()

        concurrent_executed = set(MockConnector.execution_order)

        MockConnector.reset()
        config_sequential = MagicMock(spec=RuntimeConfig)
        action1 = create_agent_action("action1", "action1")
        action2 = create_agent_action("action2", "action2")
        config_sequential.agent_actions = [action1, action2]
        config_sequential.action_execution_mode = "sequential"
        config_sequential.action_dependencies = {}

        orchestrator = ActionOrchestrator(config_sequential)
        await orchestrator.promise(actions)

        sequential_executed = set(MockConnector.execution_order)

        assert concurrent_executed == sequential_executed == {"action1", "action2"}
