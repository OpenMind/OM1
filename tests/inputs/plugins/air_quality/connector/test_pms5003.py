from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.air_quality.connector.base import AirQualityData
from inputs.plugins.air_quality.connector.pms5003 import PMS5003Connector


class TestPMS5003ConnectorInit:
    """Tests for PMS5003Connector initialization."""

    def test_default_values(self):
        connector = PMS5003Connector({})
        assert connector.port == "/dev/ttyUSB0"
        assert connector.location == "Robot"
        assert connector._serial is None

    def test_custom_values(self):
        config = {"port": "/dev/ttyAMA0", "location": "Outdoor"}
        connector = PMS5003Connector(config)
        assert connector.port == "/dev/ttyAMA0"
        assert connector.location == "Outdoor"


class TestPMS5003ConnectorConnect:
    """Tests for connect/disconnect."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        connector = PMS5003Connector({"port": "/dev/ttyUSB0"})
        with patch(
            "inputs.plugins.air_quality.connector.pms5003.serial.Serial"
        ) as mock_serial:
            mock_serial.return_value = MagicMock()
            result = await connector.connect()
        assert result is True
        assert connector._serial is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        import serial

        connector = PMS5003Connector({"port": "/dev/ttyUSB0"})
        with patch(
            "inputs.plugins.air_quality.connector.pms5003.serial.Serial",
            side_effect=serial.SerialException("Port not found"),
        ):
            result = await connector.connect()
        assert result is False
        assert connector._serial is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_serial(self):
        connector = PMS5003Connector({})
        mock_serial = MagicMock()
        mock_serial.is_open = True
        connector._serial = mock_serial

        await connector.disconnect()
        mock_serial.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        connector = PMS5003Connector({})
        await connector.disconnect()  # should not raise


class TestPMS5003ConnectorRead:
    """Tests for read()."""

    @pytest.fixture
    def connected_connector(self):
        connector = PMS5003Connector({"port": "/dev/ttyUSB0", "location": "Robot"})
        connector._serial = MagicMock()
        connector._serial.is_open = True
        return connector

    @pytest.mark.asyncio
    async def test_read_returns_none_when_not_connected(self):
        connector = PMS5003Connector({})
        result = await connector.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_success(self, connected_connector):
        # Build valid 32-byte PMS5003 frame
        # PM2.5 = 35 µg/m³ at bytes [6:8], PM10 = 60 µg/m³ at bytes [8:10]
        frame = bytearray(32)
        frame[0] = 0x42
        frame[1] = 0x4D
        frame[6] = 0x00
        frame[7] = 35  # PM2.5
        frame[8] = 0x00
        frame[9] = 60  # PM10
        checksum = sum(frame[:30]) & 0xFFFF
        frame[30] = (checksum >> 8) & 0xFF
        frame[31] = checksum & 0xFF

        with patch.object(
            connected_connector, "_read_frame", return_value=bytes(frame)
        ):
            result = await connected_connector.read()

        assert isinstance(result, AirQualityData)
        assert result.pm25 == 35.0
        assert result.pm10 == 60.0
        assert result.source == "pms5003"
        assert result.location == "Robot"
        assert result.aqi is not None

    @pytest.mark.asyncio
    async def test_read_returns_none_on_bad_frame(self, connected_connector):
        with patch.object(connected_connector, "_read_frame", return_value=None):
            result = await connected_connector.read()
        assert result is None


class TestPMS5003PM25ToAQI:
    """Tests for _pm25_to_aqi static method."""

    def test_good_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(5.0)
        assert 0 <= aqi <= 50

    def test_moderate_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(20.0)
        assert 51 <= aqi <= 100

    def test_unhealthy_sensitive_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(40.0)
        assert 101 <= aqi <= 150

    def test_unhealthy_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(100.0)
        assert 151 <= aqi <= 200

    def test_very_unhealthy_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(200.0)
        assert 201 <= aqi <= 300

    def test_hazardous_range(self):
        aqi = PMS5003Connector._pm25_to_aqi(400.0)
        assert 301 <= aqi <= 500

    def test_above_max_returns_500(self):
        aqi = PMS5003Connector._pm25_to_aqi(600.0)
        assert aqi == 500


class TestPMS5003ReadFrame:
    """Cover _read_frame internals: sync loop, checksum, short frame."""

    @pytest.fixture
    def connector(self):
        c = PMS5003Connector({"port": "/dev/ttyUSB0", "location": "Robot"})
        c._serial = MagicMock()
        return c

    def test_read_frame_returns_none_when_serial_none(self):
        connector = PMS5003Connector({})
        result = connector._read_frame()
        assert result is None

    def test_read_frame_empty_byte_returns_none(self, connector):
        connector._serial.read.return_value = b""
        result = connector._read_frame()
        assert result is None

    def test_read_frame_checksum_mismatch_returns_none(self, connector):
        frame = bytearray(32)
        frame[0] = 0x42
        frame[1] = 0x4D
        # Wrong checksum
        frame[30] = 0xFF
        frame[31] = 0xFF

        connector._serial.read.side_effect = [
            bytes([0x42]),  # sync byte 1
            bytes([0x4D]),  # sync byte 2
            bytes(frame[2:]),  # rest
        ]
        result = connector._read_frame()
        assert result is None

    def test_read_frame_short_rest_returns_none(self, connector):
        connector._serial.read.side_effect = [
            bytes([0x42]),
            bytes([0x4D]),
            bytes(5),  # too short
        ]
        result = connector._read_frame()
        assert result is None

    def test_read_frame_valid(self, connector):
        frame = bytearray(32)
        frame[0] = 0x42
        frame[1] = 0x4D
        checksum = sum(frame[:30]) & 0xFFFF
        frame[30] = (checksum >> 8) & 0xFF
        frame[31] = checksum & 0xFF

        connector._serial.read.side_effect = [
            bytes([0x42]),
            bytes([0x4D]),
            bytes(frame[2:]),
        ]
        result = connector._read_frame()
        assert result is not None
        assert len(result) == 32

    @pytest.mark.asyncio
    async def test_read_exception_returns_none(self, connector):
        with patch.object(
            connector, "_read_frame", side_effect=Exception("read error")
        ):
            result = await connector.read()
        assert result is None
