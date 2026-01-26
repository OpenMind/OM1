from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.rtk import Rtk


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.rtk.IOProvider"),
        patch("inputs.plugins.rtk.RtkProvider"),
    ):
        config = SensorConfig()
        sensor = Rtk(config=config)

        assert sensor.messages == []


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.rtk.IOProvider"),
        patch("inputs.plugins.rtk.RtkProvider") as mock_provider_class,
    ):
        mock_provider = MagicMock()
        mock_provider.data = {"lat": 37.7749, "lon": -122.4194}
        mock_provider_class.return_value = mock_provider

        config = SensorConfig()
        sensor = Rtk(config=config)

        with patch("inputs.plugins.rtk.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
            assert result == {"lat": 37.7749, "lon": -122.4194}


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.rtk.IOProvider"),
        patch("inputs.plugins.rtk.RtkProvider"),
    ):
        config = SensorConfig()
        sensor = Rtk(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456,
            message="Your precise location is 37.7749 North, 122.4194 West at 10m altitude. ",
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Precision Location" in result
        assert "37.7749 North" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
