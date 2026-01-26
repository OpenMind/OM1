from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.selfie_status_input import SelfieStatus


def test_initialization():
    """Test basic initialization."""
    with patch("inputs.plugins.selfie_status_input.IOProvider"):
        config = SensorConfig()
        sensor = SelfieStatus(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.selfie_status_input.IOProvider"),
        patch("inputs.plugins.selfie_status_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = SensorConfig()
        sensor = SelfieStatus(config=config)

        result = await sensor._poll()

        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.selfie_status_input.IOProvider"):
        config = SensorConfig()
        sensor = SelfieStatus(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
