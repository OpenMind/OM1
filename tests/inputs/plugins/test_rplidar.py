from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
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
        patch("inputs.plugins.rplidar.RPLidarProvider") as mock_rplidar,
        patch("inputs.plugins.rplidar.asyncio.sleep", new=AsyncMock()),
    ):
        mock_rplidar.return_value.lidar_string = (
            "Hello from RPLidar: objects and walls detected."
        )
        config = RPLidarConfig()
        sensor = RPLidar(config=config)

        result = await sensor._poll()
        assert result == "Hello from RPLidar: objects and walls detected."


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.rplidar.IOProvider"),
        patch("inputs.plugins.rplidar.RPLidarProvider"),
    ):
        config = RPLidarConfig()
        sensor = RPLidar(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="Wall detected at 0.5m ahead, clear path on left"
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "objects and walls" in result
        assert "Wall detected" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
