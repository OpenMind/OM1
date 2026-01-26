from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_vila import VLMVila, VLMVilaConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_vila.IOProvider"),
        patch("inputs.plugins.vlm_vila.VLMVilaProvider"),
    ):
        config = VLMVilaConfig()
        sensor = VLMVila(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.vlm_vila.IOProvider"),
        patch("inputs.plugins.vlm_vila.VLMVilaProvider"),
    ):
        config = VLMVilaConfig(
            api_key="test_key",
            base_url="wss://test.com",
            stream_base_url="https://stream.test.com",
            camera_index=1,
        )
        sensor = VLMVila(config=config)

        assert sensor.config.api_key == "test_key"
        assert sensor.config.base_url == "wss://test.com"
        assert sensor.config.stream_base_url == "https://stream.test.com"
        assert sensor.config.camera_index == 1


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_vila.IOProvider"),
        patch("inputs.plugins.vlm_vila.VLMVilaProvider"),
        patch("inputs.plugins.vlm_vila.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMVilaConfig()
        sensor = VLMVila(config=config)

        result = await sensor._poll()
        assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_vila.IOProvider"),
        patch("inputs.plugins.vlm_vila.VLMVilaProvider"),
    ):
        config = VLMVilaConfig()
        sensor = VLMVila(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="I see a person walking")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "I see a person walking" in result
        assert len(sensor.messages) == 0
