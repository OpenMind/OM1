from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_camera_vlm_cloud import (
    UnitreeGo2CameraVLMCloud,
    UnitreeGo2CameraVLMCloudConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.UnitreeCameraVLMProvider"),
    ):
        config = UnitreeGo2CameraVLMCloudConfig()
        sensor = UnitreeGo2CameraVLMCloud(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.UnitreeCameraVLMProvider"),
    ):
        config = UnitreeGo2CameraVLMCloudConfig(
            api_key="test_key",
            base_url="wss://test.com",
            stream_base_url="https://stream.test.com",
        )
        sensor = UnitreeGo2CameraVLMCloud(config=config)

        assert sensor.config.api_key == "test_key"
        assert sensor.config.base_url == "wss://test.com"
        assert sensor.config.stream_base_url == "https://stream.test.com"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.UnitreeCameraVLMProvider"),
        patch(
            "inputs.plugins.unitree_go2_camera_vlm_cloud.asyncio.sleep",
            new=AsyncMock(),
        ),
    ):
        config = UnitreeGo2CameraVLMCloudConfig()
        sensor = UnitreeGo2CameraVLMCloud(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.IOProvider"),
        patch("inputs.plugins.unitree_go2_camera_vlm_cloud.UnitreeCameraVLMProvider"),
    ):
        config = UnitreeGo2CameraVLMCloudConfig()
        sensor = UnitreeGo2CameraVLMCloud(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a quadruped robot")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a quadruped robot" in result
        assert len(sensor.messages) == 0
