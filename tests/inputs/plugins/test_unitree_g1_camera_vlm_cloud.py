from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_g1_camera_vlm_cloud import (
    UnitreeG1CameraVLMCloud,
    UnitreeG1CameraVLMCloudConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_g1_camera_vlm_cloud.IOProvider"),
        patch(
            "inputs.plugins.unitree_g1_camera_vlm_cloud.UnitreeRealSenseDevVLMProvider"
        ),
    ):
        config = UnitreeG1CameraVLMCloudConfig()
        sensor = UnitreeG1CameraVLMCloud(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.unitree_g1_camera_vlm_cloud.IOProvider"),
        patch(
            "inputs.plugins.unitree_g1_camera_vlm_cloud.UnitreeRealSenseDevVLMProvider"
        ),
    ):
        config = UnitreeG1CameraVLMCloudConfig(base_url="wss://test.com")
        sensor = UnitreeG1CameraVLMCloud(config=config)

        assert sensor.config.base_url == "wss://test.com"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_g1_camera_vlm_cloud.IOProvider"),
        patch(
            "inputs.plugins.unitree_g1_camera_vlm_cloud.UnitreeRealSenseDevVLMProvider"
        ),
        patch(
            "inputs.plugins.unitree_g1_camera_vlm_cloud.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = UnitreeG1CameraVLMCloudConfig()
        sensor = UnitreeG1CameraVLMCloud(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.unitree_g1_camera_vlm_cloud.IOProvider"),
        patch(
            "inputs.plugins.unitree_g1_camera_vlm_cloud.UnitreeRealSenseDevVLMProvider"
        ),
    ):
        config = UnitreeG1CameraVLMCloudConfig()
        sensor = UnitreeG1CameraVLMCloud(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a humanoid robot")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a humanoid robot" in result
        assert len(sensor.messages) == 0
