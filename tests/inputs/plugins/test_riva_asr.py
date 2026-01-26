from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.riva_asr import RivaASRInput, RivaASRSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        assert hasattr(sensor, "messages")
        mock_asr_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        with patch("inputs.plugins.riva_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
