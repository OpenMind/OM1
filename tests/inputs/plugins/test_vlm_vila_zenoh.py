from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_vila_zenoh import VLMVilaZenoh, VLMVilaZenohConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_vila_zenoh.IOProvider"),
        patch("inputs.plugins.vlm_vila_zenoh.VLMVilaZenohProvider"),
    ):
        config = VLMVilaZenohConfig()
        sensor = VLMVilaZenoh(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.vlm_vila_zenoh.IOProvider"),
        patch("inputs.plugins.vlm_vila_zenoh.VLMVilaZenohProvider"),
    ):
        config = VLMVilaZenohConfig(
            base_url="wss://test.com",
            topic="camera_feed",
            decode_format="H265",
        )
        sensor = VLMVilaZenoh(config=config)

        assert sensor.config.base_url == "wss://test.com"
        assert sensor.config.topic == "camera_feed"
        assert sensor.config.decode_format == "H265"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_vila_zenoh.IOProvider"),
        patch("inputs.plugins.vlm_vila_zenoh.VLMVilaZenohProvider"),
        patch("inputs.plugins.vlm_vila_zenoh.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMVilaZenohConfig()
        sensor = VLMVilaZenoh(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_vila_zenoh.IOProvider"),
        patch("inputs.plugins.vlm_vila_zenoh.VLMVilaZenohProvider"),
    ):
        config = VLMVilaZenohConfig()
        sensor = VLMVilaZenoh(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a crowded room")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a crowded room" in result
        assert len(sensor.messages) == 0
