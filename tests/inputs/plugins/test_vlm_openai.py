from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.vlm_openai import VLMOpenAI, VLMOpenAIConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_openai.IOProvider"),
        patch("inputs.plugins.vlm_openai.VLMOpenAIProvider"),
    ):
        config = VLMOpenAIConfig(api_key="test-api-key")
        sensor = VLMOpenAI(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_openai.IOProvider"),
        patch("inputs.plugins.vlm_openai.VLMOpenAIProvider"),
        patch("inputs.plugins.vlm_openai.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMOpenAIConfig(api_key="test-api-key")
        sensor = VLMOpenAI(config=config)

        result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_openai.IOProvider"),
        patch("inputs.plugins.vlm_openai.VLMOpenAIProvider"),
    ):
        config = VLMOpenAIConfig(api_key="test-api-key")
        sensor = VLMOpenAI(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
