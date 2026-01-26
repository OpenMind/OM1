"""Tests for SerialReader input plugin."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.serial_reader import SerialReader


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Heart Rate and Grip Strength"
        assert sensor.io_provider is not None
        assert sensor.ser is not None


def test_initialization_with_serial_error():
    """Test initialization when serial connection fails."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.serial.SerialException") as mock_exception,
    ):
        import serial

        mock_serial.side_effect = serial.SerialException("Port not found")
        config = SensorConfig()
        sensor = SerialReader(config=config)

        assert sensor.ser is None
        assert sensor.messages == []


@pytest.mark.asyncio
async def test_poll_with_data():
    """Test _poll when serial data is available."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
    ):
        mock_serial_instance = MagicMock()
        mock_serial_instance.readline.return_value = b"Pulse: Elevated\n"
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._poll()

        assert result == "Pulse: Elevated"


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no serial data is available."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
    ):
        mock_serial_instance = MagicMock()
        mock_serial_instance.readline.return_value = b"\n"
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_poll_with_no_serial_connection():
    """Test _poll when serial connection is None."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
        patch("inputs.plugins.serial_reader.serial.SerialException") as mock_exception,
    ):
        import serial

        mock_serial.side_effect = serial.SerialException("Port not found")
        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_pulse_data():
    """Test _raw_to_text with pulse data."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._raw_to_text("Pulse: Elevated")

        assert result is not None
        assert result.timestamp == 1234.0
        assert "pulse rate" in result.message.lower()
        assert "Elevated" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_grip_data():
    """Test _raw_to_text with grip data."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._raw_to_text("Grip: Normal")

        assert result is not None
        assert result.timestamp == 1234.0
        assert "grip strength" in result.message.lower()
        assert "Normal" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_unknown_data():
    """Test _raw_to_text with unknown data format."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._raw_to_text("Unknown: Data")

        assert result is not None
        assert result.message == "No serial data."


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = await sensor._raw_to_text(None)

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer():
    """Test that raw_to_text updates the message buffer."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        assert len(sensor.messages) == 0

        await sensor.raw_to_text("Pulse: Elevated")

        assert len(sensor.messages) == 1
        assert "pulse rate" in sensor.messages[0].message.lower()


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None


def test_formatted_latest_buffer_with_message():
    """Test formatted_latest_buffer formats message correctly."""
    with (
        patch("inputs.plugins.serial_reader.serial.Serial") as mock_serial,
        patch("inputs.plugins.serial_reader.IOProvider") as mock_io_provider,
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_io_instance = MagicMock()
        mock_io_provider.return_value = mock_io_instance

        config = SensorConfig()
        sensor = SerialReader(config=config)
        sensor.io_provider = mock_io_instance

        message = Message(timestamp=time.time(), message="Test message")
        sensor.messages.append(message)

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Heart Rate and Grip Strength" in result
        assert "Test message" in result
        assert "// START" in result
        assert "// END" in result
        mock_io_instance.add_input.assert_called_once()
        assert len(sensor.messages) == 0
