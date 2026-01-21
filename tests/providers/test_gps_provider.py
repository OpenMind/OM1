from unittest.mock import MagicMock, patch

import pytest
import serial

from providers.gps_provider import GpsProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    GpsProvider.reset()  # type: ignore
    yield

    try:
        provider = GpsProvider()
        provider.stop()
    except Exception:
        pass

    GpsProvider.reset()  # type: ignore


@pytest.fixture
def mock_serial():
    with patch("providers.gps_provider.serial.Serial") as mock_serial_class:
        mock_serial_instance = MagicMock()
        mock_serial_instance.readline.return_value = b""
        mock_serial_instance.is_open = True
        mock_serial_class.return_value = mock_serial_instance
        yield mock_serial_class, mock_serial_instance


def test_initialization(mock_serial):
    """Test GpsProvider initialization."""
    mock_serial_class, mock_serial_instance = mock_serial
    serial_port = "/dev/ttyUSB0"

    provider = GpsProvider(serial_port)

    mock_serial_class.assert_called_once_with(serial_port, 115200, timeout=1)
    mock_serial_instance.reset_input_buffer.assert_called_once()
    assert provider.lat == 0.0
    assert provider.lon == 0.0
    assert provider.alt == 0.0
    assert provider.sat == 0
    assert provider.qua == 0
    assert provider.running


def test_singleton_pattern(mock_serial):
    """Test that GpsProvider follows singleton pattern."""
    provider1 = GpsProvider("/dev/ttyUSB0")
    provider2 = GpsProvider("/dev/ttyUSB1")
    assert provider1 is provider2


def test_serial_exception_handling():
    """Test handling of serial.SerialException during initialization."""
    with patch("providers.gps_provider.serial.Serial") as mock_serial_class:
        mock_serial_class.side_effect = serial.SerialException("Port not found")

        provider = GpsProvider("/dev/invalid")

        assert provider.serial_connection is None


def test_string_to_unix_timestamp(mock_serial):
    """Test conversion of time string to Unix timestamp."""
    provider = GpsProvider("/dev/ttyUSB0")

    time_str = "2024:01:15:10:30:45:500"
    timestamp = provider.string_to_unix_timestamp(time_str)

    assert isinstance(timestamp, float)
    assert timestamp > 0


def test_stop(mock_serial):
    """Test stopping the GpsProvider."""
    provider = GpsProvider("/dev/ttyUSB0")

    provider.stop()

    assert not provider.running
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_properties(mock_serial):
    """Test data properties of GpsProvider."""
    provider = GpsProvider("/dev/ttyUSB0")

    assert provider.lat == 0.0
    assert provider.lon == 0.0
    assert provider.alt == 0.0
    assert provider.sat == 0
    assert provider.qua == 0
    assert provider.yaw_mag_0_360 == 0.0
    assert provider.yaw_mag_cardinal == ""
    assert isinstance(provider.ble_scan, list)
