from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.turtlebot4_camera_vlm_cloud import (
    TurtleBot4CameraVLMCloud,
    TurtleBot4CameraVLMCloudConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.TurtleBot4CameraVLMProvider"),
    ):
        config = TurtleBot4CameraVLMCloudConfig()
        sensor = TurtleBot4CameraVLMCloud(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.TurtleBot4CameraVLMProvider"),
    ):
        config = TurtleBot4CameraVLMCloudConfig(
            api_key="test_key",
            base_url="wss://test.com",
            stream_base_url="https://stream.test.com",
            URID="test_urid",
        )
        sensor = TurtleBot4CameraVLMCloud(config=config)

        assert sensor.config.api_key == "test_key"
        assert sensor.config.base_url == "wss://test.com"
        assert sensor.config.stream_base_url == "https://stream.test.com"
        assert sensor.config.URID == "test_urid"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.TurtleBot4CameraVLMProvider"),
        patch(
            "inputs.plugins.turtlebot4_camera_vlm_cloud.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = TurtleBot4CameraVLMCloudConfig()
        sensor = TurtleBot4CameraVLMCloud(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.turtlebot4_camera_vlm_cloud.TurtleBot4CameraVLMProvider"),
    ):
        config = TurtleBot4CameraVLMCloudConfig()
        sensor = TurtleBot4CameraVLMCloud(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a robot")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a robot" in result
        assert len(sensor.messages) == 0
