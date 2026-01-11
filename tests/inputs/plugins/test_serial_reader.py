from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import serial

from inputs.base import Message
from inputs.plugins.serial_reader import (
    DEFAULT_BAUDRATE,
    DEFAULT_SERIAL_PORT,
    DEFAULT_TIMEOUT,
    SerialReader,
    SerialReaderConfig,
)


class TestSerialReaderConfig:
    """Tests for SerialReaderConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SerialReaderConfig()

        assert config.port == DEFAULT_SERIAL_PORT
        assert config.baudrate == DEFAULT_BAUDRATE
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.descriptor == "Heart Rate and Grip Strength"

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = SerialReaderConfig(
            port="/dev/ttyACM0",
            baudrate=115200,
            timeout=2.5,
            descriptor="Custom Sensor",
        )

        assert config.port == "/dev/ttyACM0"
        assert config.baudrate == 115200
        assert config.timeout == 2.5
        assert config.descriptor == "Custom Sensor"

    def test_partial_custom_values(self):
        """Test that partial custom values work with defaults."""
        config = SerialReaderConfig(port="COM3", baudrate=19200)

        assert config.port == "COM3"
        assert config.baudrate == 19200
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.descriptor == "Heart Rate and Grip Strength"


def test_initialization_success():
    """Test successful initialization with serial connection."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SerialReaderConfig())

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
        sensor = SerialReader(config=SerialReaderConfig())

        assert sensor.ser is None


def test_initialization_with_custom_config():
    """Test initialization with custom config values."""
    mock_serial = MagicMock()

    with (
        patch(
            "inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial
        ) as mock_serial_class,
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        config = SerialReaderConfig(
            port="/dev/ttyACM0",
            baudrate=115200,
            timeout=0.5,
            descriptor="Temperature Sensor",
        )
        sensor = SerialReader(config=config)

        mock_serial_class.assert_called_once_with("/dev/ttyACM0", 115200, timeout=0.5)
        assert sensor.descriptor_for_LLM == "Temperature Sensor"


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
        sensor = SerialReader(config=SerialReaderConfig())

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
        sensor = SerialReader(config=SerialReaderConfig())

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
        sensor = SerialReader(config=SerialReaderConfig())

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
        sensor = SerialReader(config=SerialReaderConfig())

        result = await sensor._raw_to_text("Pulse: Elevated")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "The child's pulse rate is Elevated."


@pytest.mark.asyncio
async def test_raw_to_text_with_grip():
    """Test _raw_to_text with grip data."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        sensor = SerialReader(config=SerialReaderConfig())

        result = await sensor._raw_to_text("Grip: Normal")

        assert result is not None
        assert "grip strength" in result.message.lower()
        assert "Normal" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_unknown():
    """Test _raw_to_text with unknown data type."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
        patch("inputs.plugins.serial_reader.time.time", return_value=1234.0),
    ):
        sensor = SerialReader(config=SerialReaderConfig())

        result = await sensor._raw_to_text("Unknown: Data")

        assert result is not None
        assert result.message == "No serial data."


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SerialReaderConfig())

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    mock_serial = MagicMock()

    with (
        patch("inputs.plugins.serial_reader.serial.Serial", return_value=mock_serial),
        patch("inputs.plugins.serial_reader.IOProvider"),
    ):
        sensor = SerialReader(config=SerialReaderConfig())
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
        sensor = SerialReader(config=SerialReaderConfig())

        result = sensor.formatted_latest_buffer()
        assert result is None
