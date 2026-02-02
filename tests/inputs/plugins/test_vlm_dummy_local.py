from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.vlm_dummy_local import DummyVLMLocal


def test_initialization():
    """Test basic initialization."""
    with patch("inputs.plugins.vlm_dummy_local.IOProvider"):
        config = SensorConfig()
        sensor = DummyVLMLocal(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.vlm_dummy_local.IOProvider"),
        patch("inputs.plugins.vlm_dummy_local.asyncio.sleep", new=AsyncMock()),
        patch("inputs.plugins.vlm_dummy_local.Image.new") as mock_image_new,
    ):
        mock_image_new.return_value = "fake_image_data"
        config = SensorConfig()
        sensor = DummyVLMLocal(config=config)

        result = await sensor._poll()
        assert result == "fake_image_data"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.vlm_dummy_local.IOProvider"):
        config = SensorConfig()
        sensor = DummyVLMLocal(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456,
            message="DUMMY VLM - FAKE DATA - I see 42 people. Also, I see a rocket.",
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Vision" in result
        assert "DUMMY VLM" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
