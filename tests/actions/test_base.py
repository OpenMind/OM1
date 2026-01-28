import threading
import time
from dataclasses import dataclass
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface


@dataclass
class SampleInput:
    value: str


@dataclass
class SampleOutput:
    result: str


# Test implementation of Interface
@dataclass
class SampleInterface(Interface[SampleInput, SampleOutput]):
    input: SampleInput
    output: SampleOutput


# Test implementation of ActionConnector
class SampleConnector(ActionConnector[ActionConfig, SampleOutput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.last_output: Optional[SampleOutput] = None

    async def connect(self, output_interface: SampleOutput) -> None:
        self.last_output = output_interface


@pytest.fixture
def action_config():
    return ActionConfig()


@pytest.fixture
def test_connector(action_config):
    return SampleConnector(action_config)


@pytest.fixture
def agent_action(test_connector):
    return AgentAction(
        name="test_action",
        llm_label="test_llm_label",
        interface=SampleInterface,
        connector=test_connector,
        exclude_from_prompt=True,
    )


@pytest.mark.asyncio
async def test_connector_connect():
    config = ActionConfig()
    connector = SampleConnector(config)
    test_output = SampleOutput(result="test_result")

    await connector.connect(test_output)

    assert connector.last_output == test_output


@pytest.mark.asyncio
async def test_full_action_flow(agent_action):
    test_input = SampleInput(value="test_data")

    # Connect the output
    await agent_action.connector.connect(test_input)
    assert isinstance(agent_action.connector, SampleConnector)
    assert agent_action.connector.last_output == test_input


def test_action_config():
    config = ActionConfig()

    assert config is not None
    assert isinstance(config, ActionConfig)


def test_agent_action_structure(agent_action):
    assert agent_action.name == "test_action"
    assert agent_action.interface == SampleInterface
    assert isinstance(agent_action.connector, SampleConnector)


def test_sleep_without_stop_event(test_connector):
    """Test that sleep works normally when no stop event is set."""
    start_time = time.time()
    result = test_connector.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_with_stop_event_not_triggered(test_connector):
    """Test that sleep completes normally when stop event is set but not triggered."""
    stop_event = threading.Event()
    test_connector.set_stop_event(stop_event)

    start_time = time.time()
    result = test_connector.sleep(0.1)
    duration = time.time() - start_time

    assert result is True
    assert duration >= 0.1


def test_sleep_interrupted_by_stop_event(test_connector):
    """Test that sleep is interrupted when stop event is set during sleep."""
    stop_event = threading.Event()
    test_connector.set_stop_event(stop_event)

    def trigger_stop():
        time.sleep(0.05)
        stop_event.set()

    # Start a thread to trigger the stop event
    stop_thread = threading.Thread(target=trigger_stop)
    stop_thread.start()

    start_time = time.time()
    result = test_connector.sleep(1.0)
    duration = time.time() - start_time

    stop_thread.join()

    assert result is False
    assert duration < 0.2
    assert test_connector.should_stop() is True


def test_sleep_already_stopped(test_connector):
    """Test that sleep returns immediately when stop event is already set."""
    stop_event = threading.Event()
    stop_event.set()
    test_connector.set_stop_event(stop_event)

    start_time = time.time()
    result = test_connector.sleep(1.0)
    duration = time.time() - start_time

    assert result is False
    assert duration < 0.1
    assert test_connector.should_stop() is True


def test_should_stop_without_event(test_connector):
    """Test that should_stop returns False when no stop event is set."""
    assert test_connector.should_stop() is False


def test_should_stop_with_event_not_set(test_connector):
    """Test that should_stop returns False when stop event exists but is not set."""
    stop_event = threading.Event()
    test_connector.set_stop_event(stop_event)

    assert test_connector.should_stop() is False


def test_should_stop_with_event_set(test_connector):
    """Test that should_stop returns True when stop event is set."""
    stop_event = threading.Event()
    test_connector.set_stop_event(stop_event)
    stop_event.set()

    assert test_connector.should_stop() is True
