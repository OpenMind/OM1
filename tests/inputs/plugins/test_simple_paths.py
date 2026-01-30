from unittest.mock import patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.simple_paths import SimplePaths


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.SimplePathsProvider"),
    ):
        config = SensorConfig()
        sensor = SimplePaths(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_simple_paths,
    ):
        mock_simple_paths.return_value.lidar_string = "ok id=wendy"
        config = SensorConfig()
        sensor = SimplePaths(config=config)

        result = await sensor._poll()
        assert result == "ok id=wendy"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.simple_paths.IOProvider"),
        patch("inputs.plugins.simple_paths.SimplePathsProvider") as mock_rplidar,
    ):
        mock_rplidar.return_value.lidar_string = (
            "Hello from RPLidar: objects and walls detected."
        )
        config = SensorConfig()
        sensor = SimplePaths(config=config)

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
