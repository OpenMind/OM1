from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.zenoh import ZenohListener, ZenohListenerConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)

        assert sensor.messages == []
        assert isinstance(sensor.message_buffer, Queue)


@pytest.mark.asyncio
async def test_poll_with_message():
    """Test _poll with message in buffer."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
        patch("inputs.plugins.zenoh.asyncio.sleep", new=AsyncMock()),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)
        sensor.message_buffer.put("Test message")

        result = await sensor._poll()

        assert result == "Test message"


@pytest.mark.asyncio
async def test_poll_empty_buffer():
    """Test _poll with empty buffer."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
        patch("inputs.plugins.zenoh.asyncio.sleep", new=AsyncMock()),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
        patch("inputs.plugins.zenoh.time.time", return_value=1234.0),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)

        result = await sensor._raw_to_text("Test message")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "Test message"


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = ["Test message"]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.zenoh.ZenohListenerProvider"),
        patch("inputs.plugins.zenoh.IOProvider"),
    ):
        config = ZenohListenerConfig()
        sensor = ZenohListener(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
