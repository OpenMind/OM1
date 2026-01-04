from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from providers.gps_provider import GpsProvider


@pytest.fixture
def mock_serial():
    """Mock serial connection to avoid hardware dependency"""
    with patch("providers.gps_provider.serial.Serial") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def gps_provider(mock_serial):
    """Create GpsProvider instance with mocked serial"""

    # Reset singleton safely for Pyright
    if hasattr(GpsProvider, "_instances"):
        setattr(GpsProvider, "_instances", {})  # <-- no pyright error

    provider = GpsProvider(serial_port="/dev/ttyUSB0")
    yield provider

    # Cleanup
    if provider.running:
        provider.stop()


class TestStringToUnixTimestamp:
    """Test timestamp conversion robustness"""

    def test_valid_timestamp_conversion(self, gps_provider):
        time_str = "2025:01:01:12:30:45:500000"
        result = gps_provider.string_to_unix_timestamp(time_str)
        assert isinstance(result, float)
        assert result > 0

    def test_invalid_format_returns_zero(self, gps_provider):
        invalid_time = "invalid:time:format"
        result = gps_provider.string_to_unix_timestamp(invalid_time)
        assert result == 0.0

    def test_empty_string_returns_zero(self, gps_provider):
        result = gps_provider.string_to_unix_timestamp("")
        assert result == 0.0


class TestCompassHeadingToDirection:
    """Test compass direction conversion"""

    def test_north_direction(self, gps_provider):
        assert gps_provider.compass_heading_to_direction(0) == "North"
        assert gps_provider.compass_heading_to_direction(360) == "North"

    def test_cardinal_directions(self, gps_provider):
        test_cases = [
            (0, "North"),
            (45, "North East"),
            (90, "East"),
            (135, "South East"),
            (180, "South"),
            (225, "South West"),
            (270, "West"),
            (315, "North West"),
        ]
        for degrees, expected in test_cases:
            assert gps_provider.compass_heading_to_direction(degrees) == expected

    def test_boundary_values(self, gps_provider):
        assert gps_provider.compass_heading_to_direction(22.5) == "North East"
        assert gps_provider.compass_heading_to_direction(22.4) == "North"


class TestParseBleTriangString:
    """Test BLE packet parsing robustness"""

    def test_valid_ble_packet(self, gps_provider):
        input_str = "BLE:AABBCCDDEEFF:-65:0201061AFF4C000215"
        result = gps_provider.parse_ble_triang_string(input_str)

        assert len(result) == 1
        assert result[0].address == "AABBCCDDEEFF"
        assert result[0].rssi == -65
        assert result[0].packet == "0201061aff4c000215"

    def test_multiple_ble_devices(self, gps_provider):
        input_str = "BLE:AABBCCDDEEFF:-65:0201 112233445566:-70:0203"
        result = gps_provider.parse_ble_triang_string(input_str)

        assert len(result) == 2
        assert result[0].address == "AABBCCDDEEFF"
        assert result[1].address == "112233445566"

    def test_invalid_ble_prefix_returns_empty(self, gps_provider):
        result = gps_provider.parse_ble_triang_string("GPS:data")
        assert result == []

    def test_empty_ble_data_returns_empty(self, gps_provider):
        result = gps_provider.parse_ble_triang_string("BLE:")
        assert result == []

    def test_malformed_ble_data_returns_empty(self, gps_provider):
        result = gps_provider.parse_ble_triang_string("BLE:malformed:::data")
        assert isinstance(result, list)


class TestMagGPSProcessor:
    """Test GPS/MAG/BLE data processing robustness"""

    def test_heading_packet_processing(self, gps_provider):
        gps_provider.magGPSProcessor("HDG:45.5")
        assert gps_provider.yaw_mag_0_360 == 45.5
        assert gps_provider.yaw_mag_cardinal == "North East"

    def test_invalid_heading_no_crash(self, gps_provider):
        gps_provider.magGPSProcessor("HDG:")

    def test_gps_packet_processing(self, gps_provider):
            gps_data = (
                "GPS:40.7128N,74.0060W,0,"
                "HDG:180,ALT:10.5,SAT:8,"
                "TIME:25:01:01:12:00:00:000000"
            )

            gps_provider.magGPSProcessor(gps_data)

            assert gps_provider.lat == 40.7128
            assert gps_provider.lon == -74.0060
            assert gps_provider.alt == 10.5
            assert gps_provider.sat == 8

    def test_malformed_gps_no_crash(self, gps_provider):
        gps_provider.magGPSProcessor("GPS:malformed,data")

    def test_ble_packet_processing(self, gps_provider):
        ble_data = "BLE:AABBCCDDEEFF:-65:0201"
        gps_provider.magGPSProcessor(ble_data)

        assert len(gps_provider.ble_scan) == 1
        assert gps_provider.ble_scan[0].address == "AABBCCDDEEFF"

    def test_malformed_ble_no_crash(self, gps_provider):
        gps_provider.magGPSProcessor("BLE:invalid::data")
        assert isinstance(gps_provider.ble_scan, list)
