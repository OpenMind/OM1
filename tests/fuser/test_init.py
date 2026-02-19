from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from actions.orchestrator import ActionResult
from fuser import Fuser
from inputs.base import Sensor, SensorConfig
from providers.io_provider import IOProvider
from runtime.config import RuntimeConfig


@dataclass
class MockSensor(Sensor[SensorConfig, Any]):
    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return "test input"


@dataclass
class MockAction:
    name: str
    llm_label: Optional[str] = None
    exclude_from_prompt: bool = False


def create_mock_config(
    agent_actions: Optional[List[MockAction]] = None,
) -> RuntimeConfig:
    """Create a mock RuntimeConfig for testing."""
    if agent_actions is None:
        agent_actions = []

    mock_config = MagicMock(spec=RuntimeConfig)
    mock_config.system_prompt_base = "system prompt base"
    mock_config.system_governance = "system governance"
    mock_config.system_prompt_examples = "system prompt examples"
    mock_config.agent_actions = agent_actions

    return mock_config


def test_fuser_initialization():
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        assert fuser.config == config
        assert fuser.io_provider == io_provider


@patch("time.time")
def test_fuser_timestamps(mock_time):
    mock_time.return_value = 1000
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        fuser.fuse([], [])
        assert io_provider.fuser_start_time == 1000
        assert io_provider.fuser_end_time == 1000


@patch("fuser.describe_action")
def test_fuser_with_inputs_and_actions(mock_describe):
    mock_describe.return_value = "action description"
    config = create_mock_config(
        agent_actions=[MockAction("action1"), MockAction("action2")]
    )
    inputs: list[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        system_prompt = (
            "\nBASIC CONTEXT:\n"
            + config.system_prompt_base
            + "\n\nLAWS:\n"
            + config.system_governance
            + "\n\nEXAMPLES:\n"
            + config.system_prompt_examples
        )

        expected = f"{system_prompt}\n\nAVAILABLE INPUTS:\ntest input\nAVAILABLE ACTIONS:\n\naction description\n\naction description\n\n\n\nWhat will you do? Actions:"
        assert result == expected
        assert mock_describe.call_count == 2
        assert io_provider.fuser_system_prompt == system_prompt
        assert io_provider.fuser_inputs == "test input"
        assert (
            io_provider.fuser_available_actions
            == "AVAILABLE ACTIONS:\naction description\n\naction description\n\n\n\nWhat will you do? Actions:"
        )


def test_no_feedback_when_empty_promises():
    """Empty promise list should produce no feedback section in prompt."""
    config = create_mock_config()

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], [])

    assert "PREVIOUS ACTION RESULTS" not in result


def test_successful_action_feedback_in_prompt():
    """Successful action should show OK in feedback."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="Hello!", success=True),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "PREVIOUS ACTION RESULTS:" in result
    assert "- speak('Hello!'): OK" in result


def test_failed_action_feedback_in_prompt():
    """Failed action should show FAILED with error in feedback."""
    config = create_mock_config()
    results = [
        ActionResult(
            action_type="move",
            action_value="forward",
            success=False,
            error="TimeoutError: Connection timed out",
        ),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "PREVIOUS ACTION RESULTS:" in result
    assert "- move('forward'): FAILED (TimeoutError: Connection timed out)" in result


def test_mixed_feedback_in_prompt():
    """Mixed success/failure should show both."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="Hi", success=True),
        ActionResult(
            action_type="move",
            action_value="forward",
            success=False,
            error="TimeoutError",
        ),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    assert "- speak('Hi'): OK" in result
    assert "- move('forward'): FAILED (TimeoutError)" in result


def test_feedback_appears_before_available_inputs():
    """Action feedback should appear before AVAILABLE INPUTS in the prompt."""
    config = create_mock_config()
    results = [
        ActionResult(action_type="speak", action_value="test", success=True),
    ]

    with patch("fuser.IOProvider", return_value=MagicMock()):
        fuser = Fuser(config)
        result = fuser.fuse([MockSensor()], results)

    feedback_pos = result.index("PREVIOUS ACTION RESULTS")
    inputs_pos = result.index("AVAILABLE INPUTS")
    assert feedback_pos < inputs_pos
