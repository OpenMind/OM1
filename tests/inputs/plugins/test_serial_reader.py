from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import serial

from inputs.base import Message, SensorConfig
from inputs.plugins.serial_reader import SerialReader


def test_initialization_success():
    """Test successful initialization with serial connection."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SensorConfig())

        assert sensor.ser == mock_serial
        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Heart Rate and Grip Strength"


def test_initialization_serial_exception():
    """Test initialization when serial connection fails."""
    with (
        patch(
            "inputs.plugins.serial_reader.serial.Serial",
            side_effect=serial.SerialException("Port not found"),
        ),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SensorConfig())

        assert sensor.ser is None


@pytest.mark.asyncio
async def test_poll_with_data():
    """Test _poll when serial data is available."""
    mock_serial = MagicMock()
    mock_serial.in_waiting = 1
    mock_serial.readline.return_value = b"Pulse: Elevated\n"

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = await sensor._poll()

        assert result == "Pulse: Elevated"


@pytest.mark.asyncio
async def test_poll_no_data():
    """Test _poll when no serial data is available."""
    mock_serial = MagicMock()
    mock_serial.in_waiting = 0
    mock_serial.readline.return_value = b""

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_poll_with_no_serial_connection():
    """Test _poll when serial connection is None."""
    with (
        patch(
            "inputs.plugins.serial_reader.serial.Serial",
            side_effect=serial.SerialException("Port not found"),
        ),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.asyncio.sleep", new=AsyncMock()),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = await sensor._raw_to_text("Pulse: Elevated")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "The child's pulse rate is Elevated."


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SensorConfig())
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Pulse: Normal"),
            Message(timestamp=1001.0, message="Grip: Elevated"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        sensor.io_provider.add_input.assert_called()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SensorConfig())

        result = sensor.formatted_latest_buffer()
        assert result is None
