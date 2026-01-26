from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message, SensorConfig
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
        patch("inputs.plugins.selfie_status_input.IOProvider") as mock_io_provider,
        patch("inputs.plugins.selfie_status_input.asyncio.sleep", new=AsyncMock()),
    ):
        mock_rec = type(
            "obj", (object,), {"timestamp": 123.456, "input": "ok id=wendy"}
        )()
        mock_io_provider.return_value.inputs.get.return_value = mock_rec
        config = SensorConfig()
        sensor = SelfieStatus(config=config)

        result = await sensor._poll()
        assert result == "ok id=wendy"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with patch("inputs.plugins.selfie_status_input.IOProvider"):
        config = SensorConfig()
        sensor = SelfieStatus(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="ok id=wendy")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "SelfieStatus" in result
        assert "ok id=wendy" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
