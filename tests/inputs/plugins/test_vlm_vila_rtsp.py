from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_vila_rtsp import VLMVilaRTSP, VLMVilaRTSPConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_vila_rtsp.IOProvider"),
        patch("inputs.plugins.vlm_vila_rtsp.VLMVilaRTSPProvider"),
    ):
        config = VLMVilaRTSPConfig()
        sensor = VLMVilaRTSP(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.vlm_vila_rtsp.IOProvider"),
        patch("inputs.plugins.vlm_vila_rtsp.VLMVilaRTSPProvider"),
    ):
        config = VLMVilaRTSPConfig(
            base_url="wss://test.com",
            rtsp_url="rtsp://test.com:8554/camera",
            decode_format="H265",
        )
        sensor = VLMVilaRTSP(config=config)

        assert sensor.config.base_url == "wss://test.com"
        assert sensor.config.rtsp_url == "rtsp://test.com:8554/camera"
        assert sensor.config.decode_format == "H265"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_vila_rtsp.IOProvider"),
        patch("inputs.plugins.vlm_vila_rtsp.VLMVilaRTSPProvider"),
        patch("inputs.plugins.vlm_vila_rtsp.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMVilaRTSPConfig()
        sensor = VLMVilaRTSP(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_vila_rtsp.IOProvider"),
        patch("inputs.plugins.vlm_vila_rtsp.VLMVilaRTSPProvider"),
    ):
        config = VLMVilaRTSPConfig()
        sensor = VLMVilaRTSP(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a building entrance")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a building entrance" in result
        assert len(sensor.messages) == 0
