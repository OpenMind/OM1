from unittest.mock import MagicMock, patch

import pytest

from providers.rplidar_driver import RPDriver, RPLidarException


def test_rplidar_driver_initialization():
    """Test RPDriver initialization with mocked serial port."""
    with patch("providers.rplidar_driver.serial.Serial") as mock_serial:
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        driver = RPDriver(port="/dev/ttyUSB0")

        mock_serial.assert_called_once()
        assert driver is not None


def test_rplidar_driver_get_info():
    """Test getting device info."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.side_effect = [
            0,
            7,
            20,
        ]  # 0 for initial check, then sizes

        driver = RPDriver(port="/dev/ttyUSB0")

        descriptor = b"\xa5\x5a\x14\x00\x00\x00\x04"
        info_response = b"\x01\x02\x03\x04" + b"\x00" * 16
        mock_serial_instance.read.side_effect = [descriptor, info_response]

        info = driver.get_info()
        assert info is not None
        assert "model" in info
        assert "firmware" in info
        assert "hardware" in info
        assert "serial number" in info


def test_rplidar_driver_start():
    """Test starting a scan."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.side_effect = [
            0,
            7,
            3,
            7,
        ]  # For health descriptor, health data, scan descriptor

        driver = RPDriver(port="/dev/ttyUSB0")

        health_descriptor = b"\xa5\x5a\x03\x00\x00\x00\x06"
        health_response = b"\x00\x00\x00"  # Good health
        scan_descriptor = b"\xa5\x5a\x05\x00\x00\x01\x81"  # normal scan descriptor
        mock_serial_instance.read.side_effect = [
            health_descriptor,
            health_response,
            scan_descriptor,
        ]

        driver.start(scan_type="normal")
        assert driver.scanning[0] is True
        assert driver.scanning[2] == "normal"


def test_rplidar_driver_get_health_wrong_reply_length():
    """Test get_health raises exception on wrong reply length."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.return_value = 0

        driver = RPDriver(port="/dev/ttyUSB0")

        # descriptor with wrong dsize (99 instead of HEALTH_LEN=3)
        bad_descriptor = b"\xa5\x5a\x63\x00\x00\x00\x06"
        mock_serial_instance.read.side_effect = [bad_descriptor]

        with pytest.raises(RPLidarException, match="Wrong get_health reply length"):
            driver.get_health()


def test_rplidar_driver_start_wrong_scan_reply_length():
    """Test start raises exception on wrong scan reply length."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.side_effect = [0, 7, 3, 7]

        driver = RPDriver(port="/dev/ttyUSB0")

        health_descriptor = b"\xa5\x5a\x03\x00\x00\x00\x06"
        health_response = b"\x00\x00\x00"  # Good health
        # scan descriptor with wrong dsize (99 instead of 5)
        bad_scan_descriptor = b"\xa5\x5a\x63\x00\x00\x01\x81"
        mock_serial_instance.read.side_effect = [
            health_descriptor,
            health_response,
            bad_scan_descriptor,
        ]

        with pytest.raises(RPLidarException, match="Wrong start scan reply length"):
            driver.start(scan_type="normal")


def test_rplidar_driver_stop():
    """Test stopping the driver."""
    with patch("providers.rplidar_driver.serial.Serial") as mock_serial:
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        driver = RPDriver(port="/dev/ttyUSB0")
        driver.scanning = [True, 5, "normal"]  # Simulate active scan

        driver.stop()

        assert driver.scanning[0] is False
        mock_serial_instance.write.assert_called()
        mock_serial_instance.flushInput.assert_called()
