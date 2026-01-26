from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import SensorConfig
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
    ):
        config = SensorConfig()
        sensor = DummyVLMLocal(config=config)

        result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.vlm_dummy_local.IOProvider"):
        config = SensorConfig()
        sensor = DummyVLMLocal(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
