"""Tests for Gps input plugin."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.gps import Gps


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_instance.data = None
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "GPS Location"
        assert sensor.io_provider is not None
        assert sensor.gps is not None


@pytest.mark.asyncio
async def test_poll_with_data():
    """Test _poll when GPS data is available."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
        patch("inputs.plugins.gps.asyncio.sleep", new=AsyncMock()),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_instance.data = {
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "gps_qua": 5,
        }
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        result = await sensor._poll()

        assert result is not None
        assert result["gps_lat"] == 40.7128
        assert result["gps_lon"] == -74.0060


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no GPS data is available."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
        patch("inputs.plugins.gps.asyncio.sleep", new=AsyncMock()),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_instance.data = None
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_gps_data():
    """Test _raw_to_text with valid GPS data."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
        patch("inputs.plugins.gps.time.time", return_value=1234.0),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "gps_qua": 5,
        }

        result = await sensor._raw_to_text(gps_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "GPS location" in result.message.lower()
        assert "40.7128" in result.message
        assert "North" in result.message
        assert "West" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_south_east_coordinates():
    """Test _raw_to_text with south and east coordinates."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
        patch("inputs.plugins.gps.time.time", return_value=1234.0),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": -33.8688,
            "gps_lon": 151.2093,
            "gps_alt": 5.0,
            "gps_qua": 4,
        }

        result = await sensor._raw_to_text(gps_data)

        assert result is not None
        assert "South" in result.message
        assert "East" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_low_quality():
    """Test _raw_to_text with low quality GPS data."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        gps_data = {
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "gps_qua": 0,  # Low quality
        }

        result = await sensor._raw_to_text(gps_data)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        result = await sensor._raw_to_text(None)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer():
    """Test that raw_to_text updates the message buffer."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        assert len(sensor.messages) == 0

        gps_data = {
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "gps_qua": 5,
        }

        await sensor.raw_to_text(gps_data)

        assert len(sensor.messages) == 1
        assert "GPS location" in sensor.messages[0].message.lower()


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider"),
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance

        config = SensorConfig()
        sensor = Gps(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None


def test_formatted_latest_buffer_with_message():
    """Test formatted_latest_buffer formats message correctly."""
    with (
        patch("inputs.plugins.gps.GpsProvider") as mock_gps_provider,
        patch("inputs.plugins.gps.IOProvider") as mock_io_provider,
    ):
        mock_gps_instance = MagicMock()
        mock_gps_provider.return_value = mock_gps_instance
        mock_io_instance = MagicMock()
        mock_io_provider.return_value = mock_io_instance

        config = SensorConfig()
        sensor = Gps(config=config)
        sensor.io_provider = mock_io_instance

        message = Message(timestamp=time.time(), message="GPS location test")
        sensor.messages.append(message)

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: GPS Location" in result
        assert "GPS location test" in result
        assert "// START" in result
        assert "// END" in result
        mock_io_instance.add_input.assert_called_once()
        assert len(sensor.messages) == 0
