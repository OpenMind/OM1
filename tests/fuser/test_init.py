from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from fuser import Fuser
from inputs.base import Sensor, SensorConfig
from providers.io_provider import IOProvider
from runtime.single_mode.config import RuntimeConfig


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


@dataclass
class MockActionResult:
    action: str


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


def test_fuser_with_empty_finished_promises():
    """Test that empty finished_promises produces no extra section in prompt."""
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], [])

        assert "PREVIOUS ACTION RESULTS" not in result


def test_fuser_with_finished_promises():
    """Test that finished_promises are included in the fused prompt."""
    config = create_mock_config()
    io_provider = IOProvider()
    promises = [
        MockActionResult(action="forward"),
        MockActionResult(action="hello world"),
    ]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], promises)

        assert "PREVIOUS ACTION RESULTS" in result
        assert "action=forward" in result
        assert "action=hello world" in result


@patch("fuser.describe_action")
def test_fuser_action_feedback_placement(mock_describe):
    """Test that action feedback appears between inputs and available actions."""
    mock_describe.return_value = "action description"
    config = create_mock_config(agent_actions=[MockAction("move")])
    inputs: list[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()
    promises = [MockActionResult(action="forward")]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, promises)

        inputs_pos = result.index("AVAILABLE INPUTS:")
        feedback_pos = result.index("PREVIOUS ACTION RESULTS:")
        actions_pos = result.index("AVAILABLE ACTIONS:")

        assert inputs_pos < feedback_pos < actions_pos


def test_format_action_feedback_empty():
    """Test _format_action_feedback with empty list."""
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser._format_action_feedback([])

        assert result == ""


def test_format_action_feedback_with_results():
    """Test _format_action_feedback formats dataclass results correctly."""
    config = create_mock_config()
    io_provider = IOProvider()
    promises = [MockActionResult(action="turn left")]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser._format_action_feedback(promises)

        assert "PREVIOUS ACTION RESULTS:" in result
        assert "- action=turn left" in result


def test_format_action_feedback_with_non_dataclass():
    """Test _format_action_feedback handles objects without __dict__."""
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser._format_action_feedback(["plain string result"])

        assert "PREVIOUS ACTION RESULTS:" in result
        assert "- plain string result" in result
