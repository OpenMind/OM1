from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.conversation_history_input import (
    ConversationHistoryConfig,
    ConversationHistoryInput,
)


def test_initialization():
    """Test basic initialization with default config."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        assert len(sensor.messages) == 0
        assert sensor.messages.maxlen == 3  # default max_rounds
        assert sensor._last_recorded_tick == -1
        assert sensor.descriptor_for_LLM == "Conversation History"
        assert sensor._stopped is False


def test_initialization_with_custom_max_rounds():
    """Test initialization with custom max_rounds."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig(max_rounds=5)
        sensor = ConversationHistoryInput(config=config)

        assert sensor.messages.maxlen == 5


@pytest.mark.asyncio
async def test_poll_with_new_voice_input():
    """Test _poll when there's new voice input."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        mock_voice_input = MagicMock()
        mock_voice_input.input = "Hello, robot!"
        mock_voice_input.tick = 5
        mock_provider.get_input.return_value = mock_voice_input
        sensor.io_provider = mock_provider

        result = await sensor._poll()

        assert result == "Hello, robot!"
        assert sensor._last_recorded_tick == 5
        mock_provider.get_input.assert_called_once_with("Voice")


@pytest.mark.asyncio
async def test_poll_with_whitespace_input():
    """Test _poll filters out whitespace-only input."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        mock_voice_input = MagicMock()
        mock_voice_input.input = "   "  # only whitespace
        mock_voice_input.tick = 5
        mock_provider.get_input.return_value = mock_voice_input
        sensor.io_provider = mock_provider

        result = await sensor._poll()

        assert result is None
        assert sensor._last_recorded_tick == -1  # should not update


@pytest.mark.asyncio
async def test_poll_with_no_new_voice_input():
    """Test _poll when there's no new voice input."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        mock_provider.get_input.return_value = None
        sensor.io_provider = mock_provider

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_poll_with_old_tick():
    """Test _poll when tick hasn't advanced."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        sensor.io_provider = mock_provider
        sensor._last_recorded_tick = 5  # already recorded this tick

        result = await sensor._poll()

        assert result is None
        mock_provider.get_input.assert_not_called()


@pytest.mark.asyncio
async def test_poll_when_stopped():
    """Test _poll returns None when input is stopped."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)
        sensor._stopped = True

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        sensor.io_provider = mock_provider

        result = await sensor._poll()

        assert result is None
        mock_provider.get_input.assert_not_called()


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.time.time", return_value=1234.5
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        result = await sensor._raw_to_text("Test message")

        assert result is not None
        assert isinstance(result, Message)
        assert result.timestamp == 1234.5
        assert result.message == "Test message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        result = await sensor._raw_to_text(None)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_messages():
    """Test raw_to_text updates message buffer."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.time.time", return_value=1234.5
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        await sensor.raw_to_text("First message")
        await sensor.raw_to_text("Second message")

        assert len(sensor.messages) == 2
        assert sensor.messages[0].message == "First message"
        assert sensor.messages[1].message == "Second message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_does_not_update():
    """Test raw_to_text with None does not update buffer."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        await sensor.raw_to_text(None)

        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages in buffer."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        sensor.messages.append(Message(timestamp=1000.0, message="First message"))
        sensor.messages.append(Message(timestamp=1001.0, message="Second message"))
        sensor.messages.append(Message(timestamp=1002.0, message="Third message"))

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Conversation History:" in result
        assert "User: First message" in result
        assert "User: Second message" in result
        assert "User: Third message" in result


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None


def test_formatted_latest_buffer_single_message():
    """Test formatted_latest_buffer with single message."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        sensor.messages.append(Message(timestamp=1000.0, message="Only message"))

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Conversation History:" in result
        assert "User: Only message" in result


def test_sliding_window_behavior():
    """Test that messages respect max_rounds sliding window."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.time.time",
            side_effect=[1000.0, 1001.0, 1002.0, 1003.0],
        ),
    ):
        config = ConversationHistoryConfig(max_rounds=3)
        sensor = ConversationHistoryInput(config=config)

        sensor.messages.append(Message(timestamp=1000.0, message="First"))
        sensor.messages.append(Message(timestamp=1001.0, message="Second"))
        sensor.messages.append(Message(timestamp=1002.0, message="Third"))
        sensor.messages.append(Message(timestamp=1003.0, message="Fourth"))

        assert len(sensor.messages) == 3  # max_rounds
        assert sensor.messages[0].message == "Second"
        assert sensor.messages[1].message == "Third"
        assert sensor.messages[2].message == "Fourth"


def test_stop_clears_messages():
    """Test stop method clears messages and sets stopped flag."""
    with patch("inputs.plugins.conversation_history_input.IOProvider"):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        sensor.messages.append(Message(timestamp=1000.0, message="First"))
        sensor.messages.append(Message(timestamp=1001.0, message="Second"))

        assert len(sensor.messages) == 2
        assert sensor._stopped is False

        sensor.stop()

        assert len(sensor.messages) == 0
        assert sensor._stopped is True


@pytest.mark.asyncio
async def test_full_workflow():
    """Test full workflow: poll -> raw_to_text -> formatted_latest_buffer."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
        patch(
            "inputs.plugins.conversation_history_input.time.time",
            side_effect=[1000.0, 1001.0],
        ),
    ):
        config = ConversationHistoryConfig(max_rounds=2)
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 1
        mock_voice_input1 = MagicMock()
        mock_voice_input1.input = "Hello"
        mock_voice_input1.tick = 1
        mock_provider.get_input.return_value = mock_voice_input1
        sensor.io_provider = mock_provider

        raw_input1 = await sensor._poll()
        await sensor.raw_to_text(raw_input1)

        mock_provider.tick_counter = 2
        mock_voice_input2 = MagicMock()
        mock_voice_input2.input = "How are you?"
        mock_voice_input2.tick = 2
        mock_provider.get_input.return_value = mock_voice_input2

        raw_input2 = await sensor._poll()
        await sensor.raw_to_text(raw_input2)

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "User: Hello" in result
        assert "User: How are you?" in result
        assert len(sensor.messages) == 2


@pytest.mark.asyncio
async def test_poll_with_input_object_no_input_field():
    """Test _poll handles voice input with missing or empty input field."""
    with (
        patch("inputs.plugins.conversation_history_input.IOProvider"),
        patch(
            "inputs.plugins.conversation_history_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = ConversationHistoryConfig()
        sensor = ConversationHistoryInput(config=config)

        mock_provider = MagicMock()
        mock_provider.tick_counter = 5
        mock_voice_input = MagicMock()
        mock_voice_input.input = None
        mock_voice_input.tick = 5
        mock_provider.get_input.return_value = mock_voice_input
        sensor.io_provider = mock_provider

        result = await sensor._poll()

        assert result is None
