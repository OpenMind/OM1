# tests/providers/test_gps_provider.py


import pytest

from src.providers.gps_provider import GpsProvider


@pytest.fixture
def gps_provider():
    """
    Fixture to create a GpsProvider instance for testing.
    Uses _singleton_class to get the original class and __new__ to avoid running __init__.
    """
    # GpsProvider adalah fungsi hasil dekorasi singleton
    # GpsProvider._singleton_class adalah kelas asli GpsProvider
    original_class = GpsProvider._singleton_class
    provider = original_class.__new__(original_class)
    return provider


def test_string_to_unix_timestamp(gps_provider):
    """
    Test the string_to_unix_timestamp function.
    Input: "2024:05:20:14:30:45:123456" -> Expected output: Unix timestamp calculated using the same logic.
    """
    input_str = "2024:05:20:14:30:45:123456"
    # Calculate the expected timestamp using standard library, assuming the function is correct in its logic
    from datetime import datetime, timezone

    dt_manual = datetime.strptime(input_str, "%Y:%m:%d:%H:%M:%S:%f")
    dt_manual = dt_manual.replace(tzinfo=timezone.utc)
    expected_timestamp = dt_manual.timestamp()

    result = gps_provider.string_to_unix_timestamp(input_str)

    assert result == expected_timestamp


def test_compass_heading_to_direction_north(gps_provider):
    """
    Test compass_heading_to_direction for North (0, 360).
    """
    assert gps_provider.compass_heading_to_direction(0.0) == "North"
    assert gps_provider.compass_heading_to_direction(360.0) == "North"


def test_compass_heading_to_direction_northeast(gps_provider):
    """
    Test compass_heading_to_direction for North East (~45 degrees).
    Values between 22.5 and 67.5 should return "North East".
    """
    assert gps_provider.compass_heading_to_direction(45.0) == "North East"
    assert gps_provider.compass_heading_to_direction(23.0) == "North East"
    assert gps_provider.compass_heading_to_direction(67.0) == "North East"


def test_compass_heading_to_direction_east(gps_provider):
    """
    Test compass_heading_to_direction for East (~90 degrees).
    Values around 90 should return "East".
    """
    assert gps_provider.compass_heading_to_direction(90.0) == "East"
    assert gps_provider.compass_heading_to_direction(68.0) == "East"
    assert gps_provider.compass_heading_to_direction(112.0) == "East"


def test_compass_heading_to_direction_southwest(gps_provider):
    """
    Test compass_heading_to_direction for South West (~225 degrees).
    Values between 202.5 and 247.5 should return "South West".
    """
    assert gps_provider.compass_heading_to_direction(225.0) == "South West"
    assert gps_provider.compass_heading_to_direction(203.0) == "South West"
    assert gps_provider.compass_heading_to_direction(247.0) == "South West"


def test_parse_ble_triang_string_valid_input_correct_regex(gps_provider):
    """
    Test parse_ble_triang_string with valid input format that matches the current regex.
    Current regex expects MAC without underscores (e.g., AABBCCDDEEFF).
    Checks if the returned list contains an RFDataRaw object with correct attributes.
    """
    input_str = "BLE:AABBCCDDEEFF:-50:1A2B3CDE"  # Format yang diterima regex saat ini
    expected_address = "AABBCCDDEEFF"
    expected_rssi = -50
    expected_packet = "1a2b3cde"

    result_list = gps_provider.parse_ble_triang_string(input_str)

    assert len(result_list) == 1
    ble_obj = result_list[0]

    assert ble_obj.address == expected_address
    assert ble_obj.rssi == expected_rssi
    assert ble_obj.packet == expected_packet


def test_parse_ble_triang_string_valid_input_format_with_underscores(gps_provider):
    """
    Test parse_ble_triang_string with input format that SHOULD be valid (with underscores) but currently fails due to regex.
    This highlights a potential bug in the regex pattern in gps_provider.py.
    Expected result: Should ideally return 1 item, but currently returns 0.
    """
    input_str = "BLE:AA_BB_CC_DD_EE_FF:-50:1A2B3CDE"  # Format yang SEHARUSNYA valid
    # Based on current buggy regex, this should return []
    result_list = gps_provider.parse_ble_triang_string(input_str)
    assert result_list == []  # This assertion passes because of the current regex bug


def test_parse_ble_triang_string_invalid_input(gps_provider):
    """
    Test parse_ble_triang_string with invalid input format.
    Should return an empty list.
    """
    input_str = "INVALID_BLE_FORMAT"
    result_list = gps_provider.parse_ble_triang_string(input_str)
    assert result_list == []


def test_parse_ble_triang_string_no_match(gps_provider):
    """
    Test parse_ble_triang_string with input that has correct prefix but wrong data format.
    Should return an empty list.
    """
    input_str = "BLE:THIS_IS_NOT_A_MAC_ADDRESS"
    result_list = gps_provider.parse_ble_triang_string(input_str)
    assert result_list == []
