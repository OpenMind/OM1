from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from fuser import Fuser
from inputs.base import Sensor, SensorConfig
from providers.io_provider import IOProvider
from providers.semantic_memory_provider import SemanticMemoryProvider
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


@patch("fuser.describe_action")
def test_fuser_injects_memories_when_enabled(mock_describe):
    """Test that relevant memories are injected into the fused prompt."""
    mock_describe.return_value = "action description"
    config = create_mock_config(agent_actions=[MockAction("action1")])
    config.mode = "test_mode"
    inputs: list[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()

    # Setup semantic memory to return memories
    SemanticMemoryProvider.reset()  # type: ignore
    memory = SemanticMemoryProvider()
    memory.enabled = True
    memory.retrieve = MagicMock(  # type: ignore
        return_value=["Input: user said hello | Response: waved back"]
    )

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

    memory.retrieve.assert_called_once_with(query="test input", mode="test_mode")
    assert "RELEVANT MEMORIES:" in result
    assert "user said hello" in result
    assert "waved back" in result

    # Cleanup
    SemanticMemoryProvider.reset()  # type: ignore


@patch("fuser.describe_action")
def test_fuser_skips_memories_when_disabled(mock_describe):
    """Test that no memory retrieval happens when semantic memory is disabled."""
    mock_describe.return_value = "action description"
    config = create_mock_config(agent_actions=[MockAction("action1")])
    inputs: list[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()

    SemanticMemoryProvider.reset()  # type: ignore
    memory = SemanticMemoryProvider()
    memory.enabled = False

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = fuser.fuse(inputs, [])

    assert "RELEVANT MEMORIES:" not in result

    SemanticMemoryProvider.reset()  # type: ignore
