import pytest

from src.providers.rtk_provider import RtkProvider


@pytest.fixture
def rtk_provider():
    """
    Fixture to create an RtkProvider instance for testing.
    """
    original_class = RtkProvider._singleton_class  # type: ignore
    provider = original_class.__new__(original_class)
    return provider


def test_get_latest_gngga_message_single_valid(rtk_provider):
    """Test with a single valid GNGGA message."""
    nmea_data = "$GNGGA,123456.000,1234.5678,N,00123.4567,E,1,8,1.0,100.0,M,47.0,M,,*62"
    expected = "$GNGGA,123456.000,1234.5678,N,00123.4567,E,1,8,1.0,100.0,M,47.0,M,,*62"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result == expected


def test_get_latest_gngga_message_multiple_messages(rtk_provider):
    """Test with multiple GNGGA messages, should return the one with the latest time."""
    nmea_data = (
        "$GPGSA,A,3,04,05,,,,,,,,,,,3.0,3.0,3.0*02\n"
        "$GNGGA,123455.000,1234.5678,N,00123.4567,E,1,8,1.0,100.0,M,47.0,M,,*63\n"  # Earlier time
        "$GNRMC,123455.000,A,1234.5678,N,00123.4567,E,0.0,0.0,010124,,,A,V*32\n"
        "$GNGGA,123457.000,1234.5678,N,00123.4567,E,1,8,1.0,101.0,M,47.0,M,,*60\n"  # Later time
        "$GNVTG,0.0,T,,M,0.0,N,0.0,K,A*26\n"
    )
    expected = "$GNGGA,123457.000,1234.5678,N,00123.4567,E,1,8,1.0,101.0,M,47.0,M,,*60"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result == expected


def test_get_latest_gngga_message_multiple_messages_same_time(rtk_provider):
    """Test with multiple GNGGA messages having the same time, should return the first one encountered."""
    nmea_data = (
        "$GNGGA,123456.000,1234.5678,N,00123.4567,E,1,8,1.0,100.0,M,47.0,M,,*62\n"  # First one encountered, same time
        "Some other data\n"
        "$GNGGA,123456.000,9999.9999,S,00999.9999,W,0,0,0.0,200.0,M,47.0,M,,*78\n"  # Second one, same time
    )
    expected = "$GNGGA,123456.000,1234.5678,N,00123.4567,E,1,8,1.0,100.0,M,47.0,M,,*62"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result == expected


def test_get_latest_gngga_message_no_gngga(rtk_provider):
    """Test with data that contains no GNGGA messages."""
    nmea_data = "$GNRMC,123456.000,A,1234.5678,N,00123.4567,E,0.0,0.0,010124,,,A,V*32\n$GPGSA,A,3,04,05,,,,,,,,,,,3.0,3.0,3.0*02"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result is None


def test_get_latest_gngga_message_empty_string(rtk_provider):
    """Test with an empty string."""
    nmea_data = ""
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result is None


def test_get_latest_gngga_message_invalid_checksums(rtk_provider):
    """Test with messages having invalid checksums, should still find the latest valid-looking one."""
    # Note: These checksums are intentionally incorrect for the data.
    # The regex doesn't validate checksums, only structure.
    nmea_data = (
        "$GNGGA,123454.000,1234.5678,N,00123.4567,E,1,8,1.0,99.0,M,47.0,M,,*FF\n"  # Earlier time
        "$GNGGA,123458.000,1234.5678,N,00123.4567,E,1,8,1.0,102.0,M,47.0,M,,*AA\n"  # Later time
    )
    expected = "$GNGGA,123458.000,1234.5678,N,00123.4567,E,1,8,1.0,102.0,M,47.0,M,,*AA"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result == expected


def test_get_latest_gngga_message_malformed_time_field(rtk_provider):
    """Test with a GNGGA message having a malformed time field, should skip it."""
    nmea_data = (
        "$GNGGA,123453.000,1234.5678,N,00123.4567,E,1,8,1.0,98.0,M,47.0,M,,*65\n"  # Valid earlier time
        "$GNGGA,NOTATIME,1234.5678,N,00123.4567,E,1,8,1.0,99.0,M,47.0,M,,*66\n"  # Malformed time
        "$GNGGA,123459.000,1234.5678,N,00123.4567,E,1,8,1.0,103.0,M,47.0,M,,*67\n"  # Valid later time
    )
    expected = "$GNGGA,123459.000,1234.5678,N,00123.4567,E,1,8,1.0,103.0,M,47.0,M,,*67"
    result = rtk_provider.get_latest_gngga_message(nmea_data)
    assert result == expected
