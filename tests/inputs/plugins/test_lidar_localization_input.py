from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.lidar_localization_input import LidarLocalizationInput


def test_initialization():
    """Test basic initialization."""
    with patch("inputs.plugins.lidar_localization_input.IOProvider"):
        config = SensorConfig()
        sensor = LidarLocalizationInput(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.lidar_localization_input.IOProvider"),
        patch("inputs.plugins.lidar_localization_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = SensorConfig()
        sensor = LidarLocalizationInput(config=config)

        result = await sensor._poll()
        assert (
            result
            == "NOT LOCALIZED: Robot position uncertain. DO NOT attempt navigation until localized."
        )


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.lidar_localization_input.IOProvider"):
        config = SensorConfig()
        sensor = LidarLocalizationInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="LOCALIZED: Robot position is confirmed."
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Robot localization status" in result
        assert "LOCALIZED: Robot position is confirmed." in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
