from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.ubtech_camera_vlm_input import (
    UbtechCameraVLMInput,
    UbtechCameraVLMSensorConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
    ):
        config = UbtechCameraVLMSensorConfig()
        sensor = UbtechCameraVLMInput(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
    ):
        config = UbtechCameraVLMSensorConfig(
            robot_ip="192.168.1.100", base_url="wss://test.com"
        )
        sensor = UbtechCameraVLMInput(config=config)

        assert sensor.config.robot_ip == "192.168.1.100"
        assert sensor.config.base_url == "wss://test.com"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = UbtechCameraVLMSensorConfig()
        sensor = UbtechCameraVLMInput(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
    ):
        config = UbtechCameraVLMSensorConfig()
        sensor = UbtechCameraVLMInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="Camera vision data")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "Camera vision data" in result
        assert len(sensor.messages) == 0
