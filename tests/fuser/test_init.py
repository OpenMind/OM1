from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

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


@dataclass
class MockSensorNone(Sensor[SensorConfig, Any]):
    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return None


@dataclass
class MockSensorUniversalLaws(Sensor[SensorConfig, Any]):
    def __init__(self):
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self):
        return "Universal Laws: Rule 1, Rule 2"


@patch("fuser.describe_action")
def test_fuse_skips_governance_when_universal_laws_in_inputs(mock_describe):
    """Test that local governance rules are omitted when blockchain Universal Laws are present."""
    mock_describe.return_value = "action description"
    config = create_mock_config(
        agent_actions=[MockAction("action1")],
    )
    inputs: list[Sensor[Any, Any]] = [MockSensorUniversalLaws()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert "LAWS:\nsystem governance" not in result
        assert "Universal Laws: Rule 1, Rule 2" in result


@patch("fuser.describe_action")
def test_fuse_includes_governance_when_no_universal_laws(mock_describe):
    """Test that local governance rules are included when no Universal Laws in inputs."""
    mock_describe.return_value = "action description"
    config = create_mock_config(
        agent_actions=[MockAction("action1")],
    )
    inputs: list[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert "LAWS:\nsystem governance" in result


def test_fuse_skips_examples_when_empty():
    """Test that EXAMPLES section is omitted when system_prompt_examples is empty."""
    config = create_mock_config()
    config.system_prompt_examples = ""
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], [])

        assert "EXAMPLES:" not in result
        assert "BASIC CONTEXT:" in result
        assert "LAWS:" in result


def test_fuse_filters_none_inputs():
    """Test that None values from sensor inputs are filtered out of the fused prompt."""
    config = create_mock_config()
    inputs: list[Sensor[Any, Any]] = [MockSensor(), MockSensorNone(), MockSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

        assert "None" not in result
        assert "test input test input" in result


@patch("fuser.describe_action")
def test_fuse_skips_excluded_actions(mock_describe):
    """Test that actions with exclude_from_prompt=True are not included in prompt."""
    mock_describe.side_effect = lambda name, label, exclude: (
        None if exclude else f"{label} description"
    )
    config = create_mock_config(
        agent_actions=[
            MockAction("visible", llm_label="visible", exclude_from_prompt=False),
            MockAction("hidden", llm_label="hidden", exclude_from_prompt=True),
        ],
    )
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], [])

        assert "visible description" in result
        assert "hidden description" not in result
        assert mock_describe.call_count == 2


def test_fuse_with_empty_inputs():
    """Test fuse with no sensor inputs produces valid prompt structure."""
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse([], [])

        assert "BASIC CONTEXT:" in result
        assert "AVAILABLE INPUTS:" in result
        assert "AVAILABLE ACTIONS:" in result
        assert "What will you do? Actions:" in result
        assert io_provider.fuser_inputs == ""
