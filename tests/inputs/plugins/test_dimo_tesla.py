from unittest.mock import AsyncMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.dimo_tesla import DIMOTesla, DIMOTeslaConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.dimo_tesla.IOProvider"),
        patch("inputs.plugins.dimo_tesla.DIMO"),
    ):
        config = DIMOTeslaConfig()
        sensor = DIMOTesla(config=config)

        assert sensor.messages == []


def test_initialization_with_credentials():
    """Test initialization with credentials."""
    with (
        patch("inputs.plugins.dimo_tesla.IOProvider"),
        patch("inputs.plugins.dimo_tesla.DIMO"),
    ):
        config = DIMOTeslaConfig(
            client_id="test_client",
            domain="test.com",
            private_key="test_key",
            token_id=123,
        )
        sensor = DIMOTesla(config=config)

        assert sensor.config.client_id == "test_client"
        assert sensor.config.domain == "test.com"
        assert sensor.config.private_key == "test_key"
        assert sensor.config.token_id == 123


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.dimo_tesla.IOProvider"),
        patch("inputs.plugins.dimo_tesla.DIMO") as mock_dimo,
        patch("inputs.plugins.dimo_tesla.time.time", return_value=1000.0),
    ):
        config = DIMOTeslaConfig()
        sensor = DIMOTesla(config=config)

        sensor.vehicle_jwt = "test_jwt"
        sensor.vehicle_jwt_expires = 2000.0
        sensor.token_id = 123

        mock_dimo_instance = mock_dimo.return_value
        mock_dimo_instance.query.return_value = {
            "data": {
                "signalsLatest": {
                    "powertrainTransmissionTravelledDistance": {"value": 1000},
                    "exteriorAirTemperature": {"value": 20},
                    "speed": {"value": 60},
                    "powertrainRange": {"value": 300},
                    "currentLocationLatitude": {"value": 37.7749},
                    "currentLocationLongitude": {"value": -122.4194},
                }
            }
        }
        sensor.dimo = mock_dimo_instance

        with patch("inputs.plugins.dimo_tesla.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert "Powertrain Transmission Travelled Distance: 1000 km" in result
        assert "Speed: 60 km/h" in result


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.dimo_tesla.IOProvider"),
        patch("inputs.plugins.dimo_tesla.DIMO"),
    ):
        config = DIMOTeslaConfig()
        sensor = DIMOTesla(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="Tesla status update")
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "Tesla status update" in result
        assert len(sensor.messages) == 1
