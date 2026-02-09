from queue import Queue
from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.google_asr import GoogleASRInput, GoogleASRSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        assert hasattr(sensor, "messages")
        assert isinstance(sensor.message_buffer, Queue)


@pytest.mark.asyncio
async def test_poll_with_message():
    """Test _poll with message in buffer."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.message_buffer.put("Test speech")

        with patch("inputs.plugins.google_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == "Test speech"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="hello world how are you")
        sensor.messages = []  # type: ignore
        sensor.messages.append(test_message)  # type: ignore

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Voice" in result
        assert "hello world how are you" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
