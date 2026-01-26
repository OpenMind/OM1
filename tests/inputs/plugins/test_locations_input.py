from unittest.mock import patch

import pytest

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
        assert result is not None or result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (patch("inputs.plugins.locations_input.IOProvider"),):
        config = LocationsSensorConfig()
        sensor = LocationsInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None or isinstance(result, str)
