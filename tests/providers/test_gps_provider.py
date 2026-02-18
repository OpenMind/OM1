from unittest.mock import MagicMock, patch

import pytest
import serial

from providers.fabric_map_provider import RFDataRaw
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


@pytest.mark.parametrize(
    "degrees, expected",
    [
        (0.0, "North"),
        (22.4, "North"),
        (22.5, "North East"),
        (45.0, "North East"),
        (90.0, "East"),
        (135.0, "South East"),
        (180.0, "South"),
        (225.0, "South West"),
        (270.0, "West"),
        (315.0, "North West"),
        (359.9, "North"),
        (360.0, "North"),
        (382.5, "North East"),
    ],
)
def test_compass_heading_to_direction(mock_serial, degrees, expected):
    """Test compass heading to cardinal direction conversion."""
    provider = GpsProvider("/dev/ttyUSB0")
    assert provider.compass_heading_to_direction(degrees) == expected


def test_parse_ble_triang_string_valid(mock_serial):
    """Test parsing valid BLE triangulation string with multiple devices."""
    provider = GpsProvider("/dev/ttyUSB0")
    input_str = "BLE:AABBCCDDEEFF:-50:0102 112233445566:-70:ab"
    result = provider.parse_ble_triang_string(input_str)

    assert len(result) == 2
    assert isinstance(result[0], RFDataRaw)
    assert result[0].address == "AABBCCDDEEFF"
    assert result[0].rssi == -50
    assert result[0].packet == "0102"
    assert result[1].address == "112233445566"
    assert result[1].rssi == -70
    assert result[1].packet == "ab"


def test_parse_ble_triang_string_no_prefix(mock_serial):
    """Test parsing string without BLE: prefix returns empty list."""
    provider = GpsProvider("/dev/ttyUSB0")
    result = provider.parse_ble_triang_string("AABBCCDDEEFF:-50:0102")
    assert result == []


def test_parse_ble_triang_string_empty(mock_serial):
    """Test parsing empty BLE string returns empty list."""
    provider = GpsProvider("/dev/ttyUSB0")
    result = provider.parse_ble_triang_string("BLE:")
    assert result == []


def test_parse_ble_triang_string_invalid_format(mock_serial):
    """Test parsing BLE string with invalid format returns empty list."""
    provider = GpsProvider("/dev/ttyUSB0")
    result = provider.parse_ble_triang_string("BLE:not-valid-data")
    assert result == []
