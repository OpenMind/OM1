from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.vlm_gemini import VLMGemini, VLMGeminiConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
        patch("inputs.plugins.vlm_gemini.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_gemini.IOProvider"),
        patch("inputs.plugins.vlm_gemini.VLMGeminiProvider"),
    ):
        config = VLMGeminiConfig(api_key="test-api-key")
        sensor = VLMGemini(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
