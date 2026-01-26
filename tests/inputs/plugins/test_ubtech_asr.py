from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.ubtech_asr import UbtechASRInput, UbtechASRSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.ubtech_asr.IOProvider"),
        patch("inputs.plugins.ubtech_asr.UbtechASRProvider"),
        patch("inputs.plugins.ubtech_asr.SleepTickerProvider"),
    ):
        config = UbtechASRSensorConfig()
        sensor = UbtechASRInput(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.ubtech_asr.IOProvider"),
        patch("inputs.plugins.ubtech_asr.UbtechASRProvider"),
        patch("inputs.plugins.ubtech_asr.SleepTickerProvider"),
        patch("inputs.plugins.ubtech_asr.asyncio.sleep", new=AsyncMock()),
    ):
        config = UbtechASRSensorConfig()
        sensor = UbtechASRInput(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.ubtech_asr.IOProvider"),
        patch("inputs.plugins.ubtech_asr.UbtechASRProvider"),
        patch("inputs.plugins.ubtech_asr.SleepTickerProvider"),
    ):
        config = UbtechASRSensorConfig()
        sensor = UbtechASRInput(config=config)

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
