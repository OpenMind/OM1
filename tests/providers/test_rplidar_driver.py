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


def test_read_response_timeout():
    """Test that _read_response raises RPLidarException on timeout."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.return_value = 0

        driver = RPDriver(port="/dev/ttyUSB0")

        with pytest.raises(RPLidarException, match="Timeout waiting for response"):
            driver._read_response(10, timeout=0.01)


def test_read_response_device_disconnect():
    """Test that _read_response raises RPLidarException when device is disconnected."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.side_effect = OSError("device disconnected")

        driver = RPDriver(port="/dev/ttyUSB0")

        with pytest.raises(RPLidarException, match="Serial device disconnected"):
            driver._read_response(10)


def test_read_response_success():
    """Test that _read_response returns data when bytes are available."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.side_effect = [0, 0, 5]
        mock_serial_instance.read.return_value = b"\x01\x02\x03\x04\x05"

        driver = RPDriver(port="/dev/ttyUSB0")
        data = driver._read_response(5)

        assert data == b"\x01\x02\x03\x04\x05"
        mock_serial_instance.read.assert_called_once_with(5)


def test_read_response_zero_dsize():
    """Test _read_response with dsize=0 returns immediately."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.return_value = 0
        mock_serial_instance.read.return_value = b""

        driver = RPDriver(port="/dev/ttyUSB0")
        data = driver._read_response(0)

        assert data == b""


def test_read_response_zero_timeout():
    """Test _read_response with timeout=0 raises immediately if data not ready."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.return_value = 0

        driver = RPDriver(port="/dev/ttyUSB0")

        with pytest.raises(RPLidarException, match="Timeout waiting for response"):
            driver._read_response(10, timeout=0)


def test_read_response_negative_timeout():
    """Test _read_response with negative timeout raises immediately."""
    with (
        patch("providers.rplidar_driver.serial.Serial") as mock_serial,
        patch("providers.rplidar_driver.time.sleep"),
    ):
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance
        mock_serial_instance.inWaiting.return_value = 0

        driver = RPDriver(port="/dev/ttyUSB0")

        with pytest.raises(RPLidarException, match="Timeout waiting for response"):
            driver._read_response(10, timeout=-1.0)

