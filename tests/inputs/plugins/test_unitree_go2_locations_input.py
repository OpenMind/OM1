from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_locations_input import (
    UnitreeGo2LocationsInput,
    UnitreeGo2LocationsSensorConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_go2_locations_input.IOProvider"),
        patch("inputs.plugins.unitree_go2_locations_input.UnitreeGo2LocationsProvider"),
    ):
        config = UnitreeGo2LocationsSensorConfig()
        sensor = UnitreeGo2LocationsInput(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.unitree_go2_locations_input.IOProvider"),
        patch("inputs.plugins.unitree_go2_locations_input.UnitreeGo2LocationsProvider"),
    ):
        config = UnitreeGo2LocationsSensorConfig(
            base_url="http://test.com/locations",
            timeout=10,
            refresh_interval=60,
        )
        sensor = UnitreeGo2LocationsInput(config=config)

        assert sensor.config.base_url == "http://test.com/locations"
        assert sensor.config.timeout == 10
        assert sensor.config.refresh_interval == 60


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_go2_locations_input.IOProvider"),
        patch(
            "inputs.plugins.unitree_go2_locations_input.UnitreeGo2LocationsProvider"
        ) as mock_provider,
        patch(
            "inputs.plugins.unitree_go2_locations_input.asyncio.sleep", new=AsyncMock()
        ),
    ):
        config = UnitreeGo2LocationsSensorConfig()
        sensor = UnitreeGo2LocationsInput(config=config)

        mock_provider_instance = mock_provider.return_value
        mock_provider_instance.get_all_locations.return_value = {
            "kitchen": {"name": "Kitchen"},
            "living_room": {"name": "Living Room"},
        }
        sensor.locations_provider = mock_provider_instance

        result = await sensor._poll()
        assert result == "Kitchen\nLiving Room"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.unitree_go2_locations_input.IOProvider"),
        patch("inputs.plugins.unitree_go2_locations_input.UnitreeGo2LocationsProvider"),
    ):
        config = UnitreeGo2LocationsSensorConfig()
        sensor = UnitreeGo2LocationsInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="Available locations: Bedroom, Hallway"
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "Available locations" in result
        assert len(sensor.messages) == 0
