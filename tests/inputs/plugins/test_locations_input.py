from unittest.mock import patch

import pytest

from inputs.base import Message
from inputs.plugins.locations_input import LocationsInput, LocationsSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (patch("inputs.plugins.locations_input.IOProvider"),):
        config = LocationsSensorConfig()
        sensor = LocationsInput(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (patch("inputs.plugins.locations_input.IOProvider"),):
        config = LocationsSensorConfig()
        sensor = LocationsInput(config=config)

        result = await sensor._poll()
        assert result == ""


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (patch("inputs.plugins.locations_input.IOProvider"),):
        config = LocationsSensorConfig()
        sensor = LocationsInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="Home (x:1.00 y:2.00)\nOffice (x:5.00 y:6.00)"
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "These are the saved locations" in result
        assert "Home (x:1.00 y:2.00)" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
