from dataclasses import dataclass
from typing import Any, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fuser import Fuser
from fuser.knowledge_base.base_retriever import Document
from inputs.base import Sensor, SensorConfig
from providers.io_provider import IOProvider
from runtime.config import RuntimeConfig


@pytest.fixture(autouse=True)
def reset_io_provider():
    """Reset the IOProvider singleton before each test to prevent state pollution."""
    IOProvider.reset()  # type: ignore
    yield
    IOProvider.reset()  # type: ignore


class MockSensor(Sensor[SensorConfig, Any]):
    def __init__(self) -> None:
        super().__init__(SensorConfig())

    def formatted_latest_buffer(self) -> str:
        return "test input"


@dataclass
class MockAction:
    name: str
    llm_label: Optional[str] = None
    exclude_from_prompt: bool = False


def create_mock_config(
    agent_actions: Optional[List[MockAction]] = None,
    knowledge_base: Optional[dict] = None,
) -> RuntimeConfig:
    """Create a mock RuntimeConfig for testing."""
    if agent_actions is None:
        agent_actions = []

    mock_config = MagicMock(spec=RuntimeConfig)
    mock_config.system_prompt_base = "system prompt base"
    mock_config.system_governance = "system governance"
    mock_config.system_prompt_examples = "system prompt examples"
    mock_config.agent_actions = agent_actions
    mock_config.knowledge_base = knowledge_base

    return mock_config


def test_fuser_initialization():
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        assert fuser.config == config
        assert fuser.io_provider == io_provider
        assert fuser.knowledge_base is None


@patch("time.time")
@pytest.mark.asyncio
async def test_fuser_timestamps(mock_time):
    mock_time.return_value = 1000
    config = create_mock_config()
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        await fuser.fuse([], [])
        assert io_provider.fuser_start_time == 1000
        assert io_provider.fuser_end_time == 1000


@patch("fuser.describe_action")
@pytest.mark.asyncio
async def test_fuser_with_inputs_and_actions(mock_describe):
    mock_describe.return_value = "action description"
    config = create_mock_config(
        agent_actions=[MockAction("action1"), MockAction("action2")]
    )
    inputs: Sequence[Sensor[Any, Any]] = [MockSensor()]
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = await fuser.fuse(inputs, [])

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


@pytest.mark.asyncio
async def test_fuser_initialization_with_knowledge_base():
    """Test that Fuser properly initializes with knowledge_base config."""
    kb_config = {
        "knowledge_base": "test_kb",
        "knowledge_base_root": "/tmp/kb",
        "base_url": "http://localhost:8100",
    }
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    mock_kb = MagicMock()
    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb) as mock_kb_class,
    ):
        fuser = Fuser(config)
        assert fuser.knowledge_base == mock_kb
        mock_kb_class.assert_called_once_with(**kb_config)


@pytest.mark.asyncio
async def test_fuser_initialization_with_invalid_knowledge_base():
    """Test that Fuser handles invalid knowledge_base config gracefully."""
    kb_config = {"invalid_param": "value"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", side_effect=Exception("Invalid config")),
    ):
        fuser = Fuser(config)
        assert fuser.knowledge_base is None


@pytest.mark.asyncio
async def test_fuser_without_knowledge_base():
    """Test that Fuser works without knowledge_base config."""
    config = create_mock_config(knowledge_base=None)
    io_provider = IOProvider()

    with patch("fuser.IOProvider", return_value=io_provider):
        fuser = Fuser(config)
        result = await fuser.fuse([], [])

        assert fuser.knowledge_base is None
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_no_voice_input():
    """Test that knowledge base is not queried when there's no voice input."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock()

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse([], [])

        mock_kb.query.assert_not_called()
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_and_voice_input():
    """Test that knowledge base is queried when voice input is available."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    voice_input = Mock()
    voice_input.input = "What is the capital of France?"
    voice_input.tick = 1
    io_provider.increment_tick()
    io_provider.get_input = Mock(return_value=voice_input)

    mock_doc1 = Document(
        text="Paris is the capital of France.",
        metadata={"source": "geography.txt", "chunk_id": 0},
        score=0.95,
    )
    mock_doc2 = Document(
        text="France is a country in Europe.",
        metadata={"source": "geography.txt", "chunk_id": 1},
        score=0.85,
    )
    mock_kb = MagicMock()
    mock_kb.query = AsyncMock(return_value=[mock_doc1, mock_doc2])
    mock_kb.format_context = Mock(
        return_value="[1] Source: geography.txt (chunk 0) | Score: 0.950\nParis is the capital of France.\n"
    )

    inputs: Sequence[Sensor[Any, Any]] = [MockSensor()]

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse(inputs, [])

        mock_kb.query.assert_called_once_with("What is the capital of France?", top_k=3)
        mock_kb.format_context.assert_called_once_with(
            [mock_doc1, mock_doc2], max_chars=1500
        )
        assert "KNOWLEDGE BASE:" in result
        assert "Paris is the capital of France." in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_empty_results():
    """Test that fuser handles empty knowledge base results gracefully."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    voice_input = Mock()
    voice_input.input = "Some query"
    voice_input.tick = 1
    io_provider.increment_tick()
    io_provider.get_input = Mock(return_value=voice_input)

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock(return_value=[])
    mock_kb.format_context = Mock(return_value="")

    inputs: Sequence[Sensor[Any, Any]] = [MockSensor()]

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse(inputs, [])

        mock_kb.query.assert_called_once()
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_query_error():
    """Test that fuser handles knowledge base query errors gracefully."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    # Setup voice input
    voice_input = Mock()
    voice_input.input = "Some query"
    voice_input.tick = 1
    io_provider.increment_tick()
    io_provider.get_input = Mock(return_value=voice_input)

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock(side_effect=Exception("Query failed"))

    inputs: Sequence[Sensor[Any, Any]] = [MockSensor()]

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse(inputs, [])

        mock_kb.query.assert_called_once()
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_voice_input_different_tick():
    """Test that knowledge base is not queried when voice input is from a different tick."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    voice_input = Mock()
    voice_input.input = "Some query"
    voice_input.tick = 5

    for _ in range(10):
        io_provider.increment_tick()
    io_provider.get_input = Mock(return_value=voice_input)

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock()

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse([], [])

        mock_kb.query.assert_not_called()
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_no_voice_input_object():
    """Test that knowledge base is not queried when voice input object is None."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    io_provider.get_input = Mock(return_value=None)

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock()

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse([], [])

        mock_kb.query.assert_not_called()
        assert "KNOWLEDGE BASE:" not in result


@pytest.mark.asyncio
async def test_fuser_with_knowledge_base_empty_voice_input():
    """Test that knowledge base is not queried when voice input is empty/whitespace."""
    kb_config = {"knowledge_base": "test_kb"}
    config = create_mock_config(knowledge_base=kb_config)
    io_provider = IOProvider()

    voice_input = Mock()
    voice_input.input = "   "
    voice_input.tick = 1
    io_provider.increment_tick()
    io_provider.get_input = Mock(return_value=voice_input)

    mock_kb = MagicMock()
    mock_kb.query = AsyncMock()

    with (
        patch("fuser.IOProvider", return_value=io_provider),
        patch("fuser.KnowledgeBase", return_value=mock_kb),
    ):
        fuser = Fuser(config)
        result = await fuser.fuse([], [])

        mock_kb.query.assert_not_called()
        assert "KNOWLEDGE BASE:" not in result
