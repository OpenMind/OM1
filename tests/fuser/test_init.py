from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from fuser import Fuser
from inputs.base import Sensor, SensorConfig
from providers.io_provider import IOProvider
from runtime.single_mode.config import RuntimeConfig


@dataclass
class MockSensor(Sensor[SensorConfig, Any]):
    def __init__(self, buffer_value="test input"):
        super().__init__(SensorConfig())
        self._buffer_value = buffer_value

    def formatted_latest_buffer(self):
        return self._buffer_value


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


@patch("time.time")
def test_fuser_skips_empty_inputs(mock_time):
    """Test that fuser returns None when all inputs are empty (fixes Issue #1025)."""
    mock_time.return_value = 1000
    config = create_mock_config()
    io_provider = IOProvider()

    # Test with empty string inputs
    empty_inputs: list[Sensor[Any, Any]] = [MockSensor(""), MockSensor("   ")]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(empty_inputs, [])

        # Should return None to prevent wasteful LLM calls
        assert result is None
        assert io_provider.fuser_start_time == 1000


@patch("time.time")
def test_fuser_skips_none_inputs(mock_time):
    """Test that fuser returns None when all inputs are None."""
    mock_time.return_value = 1000
    config = create_mock_config()
    io_provider = IOProvider()

    # Test with None inputs
    none_inputs: list[Sensor[Any, Any]] = [MockSensor(None), MockSensor(None)]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(none_inputs, [])

        # Should return None to prevent wasteful LLM calls
        assert result is None
        assert io_provider.fuser_start_time == 1000


@patch("fuser.describe_action")
@patch("time.time")
def test_fuser_processes_mixed_inputs(mock_time, mock_describe):
    """Test that fuser processes inputs when at least one is meaningful."""
    mock_time.return_value = 1000
    mock_describe.return_value = "action description"
    config = create_mock_config(agent_actions=[MockAction("action1")])
    io_provider = IOProvider()

    # Mix of empty and meaningful inputs
    mixed_inputs: list[Sensor[Any, Any]] = [
        MockSensor(""),
        MockSensor("meaningful input"),
        MockSensor(None),
    ]

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(mixed_inputs, [])

        # Should process the meaningful input
        assert result is not None
        assert "meaningful input" in result
        assert io_provider.fuser_inputs == "meaningful input"
