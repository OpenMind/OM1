from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.gps import Gps


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "GPS Location"


@pytest.mark.asyncio
async def test_poll_with_data():
    """Test _poll when GPS data is available."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_provider_class,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.data = {
            "gps_lat": 37.7749,
            "gps_lon": -122.4194,
            "gps_alt": 16.0,
            "gps_qua": 1,
        }
        mock_provider_class.return_value = mock_provider

        config = SensorConfig()
        sensor = Gps(config=config)

        with patch("inputs.plugins.gps.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert result["gps_lat"] == 37.7749
        assert result["gps_lon"] == -122.4194


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no GPS data available."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_provider_class,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.data = None
        mock_provider_class.return_value = mock_provider

        config = SensorConfig()
        sensor = Gps(config=config)

        with patch("inputs.plugins.gps.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_north_east_location():
    """Test _raw_to_text with valid North/East GPS coordinates."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": 37.7749,
            "gps_lon": 122.4194,
            "gps_alt": 16.0,
            "gps_qua": 1,
        }

        with patch("inputs.plugins.gps.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(gps_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "North" in result.message
        assert "East" in result.message
        assert "37.7749" in result.message
        assert "122.4194" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_south_west_location():
    """Test _raw_to_text with valid South/West GPS coordinates."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": -33.8688,
            "gps_lon": -151.2093,
            "gps_alt": 10.0,
            "gps_qua": 1,
        }

        with patch("inputs.plugins.gps.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(gps_data)

        assert result is not None
        assert "South" in result.message
        assert "West" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_poor_quality():
    """Test _raw_to_text with poor GPS quality."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": 37.7749,
            "gps_lon": 122.4194,
            "gps_alt": 16.0,
            "gps_qua": 0,  # Poor quality
        }

        result = await sensor._raw_to_text(gps_data)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="GPS: 37.7749 North, 122.4194 East"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "GPS" in result
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.gps.GpsProvider"),
        patch("inputs.plugins.gps.IOProvider"),
    ):
        config = SensorConfig()
        sensor = Gps(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
