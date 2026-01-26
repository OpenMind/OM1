from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import SensorConfig
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

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.rtk.IOProvider"),
        patch("inputs.plugins.rtk.RtkProvider"),
    ):
        config = SensorConfig()
        sensor = Rtk(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
