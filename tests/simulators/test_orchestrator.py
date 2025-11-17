import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import Mock, patch

import pytest

from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig
from simulators.base import Simulator, SimulatorConfig
from simulators.orchestrator import SimulatorOrchestrator


class MockSimulator(Simulator):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.tick_count = 0
        self.actions_received: List[Action] = []

    def tick(self):
        self.tick_count += 1

    def sim(self, actions: List[Action]):
        self.actions_received.extend(actions)


@pytest.fixture
def mock_config():
    config = Mock(spec=RuntimeConfig)
    simulator1 = MockSimulator(config=SimulatorConfig(name="sim1"))
    simulator2 = MockSimulator(config=SimulatorConfig(name="sim2"))
    config.simulators = [simulator1, simulator2]
    return config


@pytest.fixture
def orchestrator(mock_config):
    return SimulatorOrchestrator(mock_config)


@pytest.fixture
def test_action():
    """Create a test action with proper structure."""

    def _create_action(type="test", arg_value="1"):
        return Action(type=type, value=str(arg_value))

    return _create_action


@pytest.mark.asyncio
async def test_simulator_orchestrator_initialization(mock_config):
    """Test that SimulatorOrchestrator initializes correctly."""
    orchestrator = SimulatorOrchestrator(mock_config)
    assert orchestrator._config == mock_config
    assert len(orchestrator.promise_queue) == 0
    assert orchestrator._simulator_workers == 2


@pytest.mark.asyncio
async def test_start_simulators(orchestrator):
    """Test that simulators are started in separate threads."""
    try:
        future = orchestrator.start()

        assert isinstance(orchestrator._simulator_executor, ThreadPoolExecutor)
        assert orchestrator._simulator_executor._max_workers == orchestrator._simulator_workers

        assert len(orchestrator._submitted_simulators) == len(orchestrator._config.simulators)
        assert isinstance(future, asyncio.Future)

        expected_simulator_names = {bg.name for bg in orchestrator._config.simulators}
        assert orchestrator._submitted_simulators == expected_simulator_names
    finally:
        orchestrator.stop()


@pytest.mark.asyncio
async def test_promise_and_flush(mock_config):
    """Test sending actions to simulators and flushing promises."""
    # Create orchestrator with mock simulators
    orchestrator = SimulatorOrchestrator(mock_config)

    # Create a test action
    test_actions = [Action(type="test", value="1")]

    # Mock _promise_simulator to return immediately
    async def mock_promise_simulator(simulator, actions):
        simulator.sim(actions)
        return None

    orchestrator._promise_simulator = mock_promise_simulator

    # Send actions
    await orchestrator.promise(test_actions)

    # Verify promise queue length (one promise per simulator)
    assert len(orchestrator.promise_queue) == 2

    # Allow promises to complete
    await asyncio.sleep(0.1)

    # Flush promises
    done_promises, pending_promises = await orchestrator.flush_promises()

    # Verify promises were processed
    assert len(done_promises) == 2
    assert len(pending_promises) == 0

    # Verify that each simulator received the actions
    for simulator in mock_config.simulators:
        assert len(simulator.actions_received) == 1
        assert simulator.actions_received[0] == test_actions[0]


@pytest.mark.asyncio
async def test_promise_simulator(orchestrator, test_action):
    """Test sending actions to a single simulator."""
    test_actions = [test_action(type="test", arg_value="1")]
    simulator = orchestrator._config.simulators[0]

    with patch("logging.debug") as mock_logging:
        result = await orchestrator._promise_simulator(simulator, test_actions)

        mock_logging.assert_called_with(
            f"Calling simulator {simulator.name} with actions {test_actions}"
        )
        assert result is None
        assert len(simulator.actions_received) == 1
        assert simulator.actions_received[0] == test_actions[0]


@pytest.mark.asyncio
async def test_concurrent_simulator_operations(orchestrator):
    """Test that multiple simulators can operate concurrently."""
    test_actions1 = [Action(type="test1", value="1")]
    test_actions2 = [Action(type="test2", value="2")]

    try:
        # Start simulators
        orchestrator.start()

        # Send multiple actions
        await asyncio.gather(
            orchestrator.promise(test_actions1), orchestrator.promise(test_actions2)
        )

        # Flush promises
        done_promises, pending_promises = await orchestrator.flush_promises()

        # Verify all promises are completed
        assert len(done_promises) == 4  # 2 actions * 2 simulators
        assert len(pending_promises) == 0

        # Verify simulators received both actions
        for simulator in orchestrator._config.simulators:
            received_actions = simulator.actions_received
            assert len(received_actions) == 2
            assert test_actions1[0] in received_actions
            assert test_actions2[0] in received_actions
    finally:
        orchestrator.stop()
