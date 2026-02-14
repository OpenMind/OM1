import asyncio
from unittest.mock import AsyncMock, patch

import pytest

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
        assert isinstance(sensor.message_buffer, asyncio.Queue)
        assert sensor.messages == []


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
        sensor.message_buffer.put_nowait("Test speech")

        with patch("asyncio.sleep", new=AsyncMock()):
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

        sensor.messages.append("hello world how are you")

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Voice" in result
        assert result.count("hello world how are you") == 1
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_raw_to_text_none_skips_sleep_when_buffer_has_messages():
    """Test raw_to_text with None sets skip_sleep when messages exist."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.messages = ["existing message"]

        await sensor.raw_to_text(None)
        assert sensor.global_sleep_ticker_provider.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_concatenates_messages():
    """Test raw_to_text concatenates when messages already exist."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        await sensor.raw_to_text("hello")
        await sensor.raw_to_text("world")
        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "hello world"


def test_stop():
    """Test stop method stops ASR provider and closes Zenoh session."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        sensor.stop()
        sensor.asr.stop.assert_called_once()  # type: ignore[union-attr]
        sensor.session.close.assert_called_once()  # type: ignore[union-attr]


def test_stop_no_session():
    """Test stop method when session is None."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.session = None

        sensor.stop()
        sensor.asr.stop.assert_called_once()  # type: ignore[union-attr]
