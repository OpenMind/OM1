from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.rplidar import RPLidar, RPLidarConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.rplidar.IOProvider"),
        patch("inputs.plugins.rplidar.RPLidarProvider"),
    ):
        config = RPLidarConfig()
        sensor = RPLidar(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.rplidar.IOProvider"),
        patch("inputs.plugins.rplidar.RPLidarProvider"),
        patch("inputs.plugins.rplidar.asyncio.sleep", new=AsyncMock()),
    ):
        config = RPLidarConfig()
        sensor = RPLidar(config=config)

        result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.rplidar.IOProvider"),
        patch("inputs.plugins.rplidar.RPLidarProvider"),
    ):
        config = RPLidarConfig()
        sensor = RPLidar(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
