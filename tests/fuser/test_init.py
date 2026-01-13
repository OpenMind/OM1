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
    inputs: list[Sensor[Any, Any]] = [MockSensor()]  # Need valid input to test timestamps

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        fuser.fuse(inputs, [])
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


@dataclass
class MockEmptySensor(Sensor[SensorConfig, Any]):
    """Mock sensor that returns None (no input available)."""

    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return None


@dataclass
class MockEmptyStringSensor(Sensor[SensorConfig, Any]):
    """Mock sensor that returns empty string."""

    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return ""


def test_fuser_returns_none_when_no_inputs():
    """Test that fuser returns None when there are no input sensors.

    This prevents unnecessary LLM API calls when no inputs are available.
    See: https://github.com/OpenmindAGI/OM1/issues/1372
    """
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], [])

        assert result is None


def test_fuser_returns_none_when_all_inputs_are_none():
    """Test that fuser returns None when all sensors return None.

    This prevents unnecessary LLM API calls when sensors have no data.
    See: https://github.com/OpenmindAGI/OM1/issues/1372
    """
    config = create_mock_config()
    inputs: list[Sensor[Any, Any]] = [MockEmptySensor(), MockEmptySensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert result is None


def test_fuser_returns_none_when_all_inputs_are_empty_strings():
    """Test that fuser returns None when all sensors return empty strings.

    This prevents unnecessary LLM API calls when sensors return empty data.
    See: https://github.com/OpenmindAGI/OM1/issues/1372
    """
    config = create_mock_config()
    inputs: list[Sensor[Any, Any]] = [MockEmptyStringSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert result is None


def test_fuser_returns_prompt_when_at_least_one_input_has_data():
    """Test that fuser returns prompt when at least one sensor has data.

    Mixed inputs (some None, some with data) should still produce a prompt.
    """
    config = create_mock_config()
    inputs: list[Sensor[Any, Any]] = [MockEmptySensor(), MockSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert result is not None
        assert "test input" in result
