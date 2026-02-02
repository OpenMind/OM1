from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.mock_input import MockInput, MockSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        assert sensor.messages == []
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.host == "localhost"
        assert sensor.port == 8765


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig(input_name="Custom Mock", host="0.0.0.0", port=9000)
        sensor = MockInput(config=config)

        assert sensor.descriptor_for_LLM == "Custom Mock"
        assert sensor.host == "0.0.0.0"
        assert sensor.port == 9000


@pytest.mark.asyncio
async def test_poll_with_message_in_buffer():
    """Test _poll when there's a message in buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.message_buffer.put("Test message")

        result = await sensor._poll()

        assert result == "Test message"


@pytest.mark.asyncio
async def test_poll_with_empty_buffer():
    """Test _poll when buffer is empty."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.time.time", return_value=1234.0),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._raw_to_text("Test message")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "Test message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Message 1"),
            Message(timestamp=1001.0, message="Message 2"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Message 1" in result or "Message 2" in result
        sensor.io_provider.add_input.assert_called()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
