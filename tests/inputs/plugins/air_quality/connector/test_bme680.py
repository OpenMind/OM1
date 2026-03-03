from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.air_quality.connector.base import AirQualityData
from inputs.plugins.air_quality.connector.bme680 import BME680Connector


class TestBME680ConnectorInit:
    """Tests for BME680Connector initialization."""

    def test_default_values(self):
        connector = BME680Connector({})
        assert connector.i2c_address == 0x76
        assert connector.location == "Robot"
        assert connector.gas_baseline == 50000.0
        assert connector._sensor is None

    def test_custom_values(self):
        config = {
            "i2c_address": 0x77,
            "location": "Indoor",
            "gas_baseline": 80000.0,
        }
        connector = BME680Connector(config)
        assert connector.i2c_address == 0x77
        assert connector.location == "Indoor"
        assert connector.gas_baseline == 80000.0


class TestBME680ConnectorConnect:
    """Tests for connect/disconnect."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        connector = BME680Connector({})
        mock_bme680 = MagicMock()
        mock_sensor = MagicMock()
        mock_bme680.BME680.return_value = mock_sensor
        mock_bme680.OS_2X = 1
        mock_bme680.OS_4X = 2
        mock_bme680.OS_8X = 3
        mock_bme680.FILTER_SIZE_3 = 3
        mock_bme680.ENABLE_GAS_MEAS = 1

        with patch.dict("sys.modules", {"bme680": mock_bme680}):
            result = await connector.connect()

        assert result is True
        assert connector._sensor is not None

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        connector = BME680Connector({})
        with patch.dict("sys.modules", {"bme680": None}):
            result = await connector.connect()
        assert result is False
        assert connector._sensor is None

    @pytest.mark.asyncio
    async def test_connect_hardware_error(self):
        connector = BME680Connector({})
        mock_bme680 = MagicMock()
        mock_bme680.BME680.side_effect = Exception("I2C error")

        with patch.dict("sys.modules", {"bme680": mock_bme680}):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_clears_sensor(self):
        connector = BME680Connector({})
        connector._sensor = MagicMock()
        await connector.disconnect()
        assert connector._sensor is None


class TestBME680ConnectorRead:
    """Tests for read() and _read_sensor()."""

    @pytest.mark.asyncio
    async def test_read_returns_none_when_not_connected(self):
        connector = BME680Connector({})
        result = await connector.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_success(self):
        connector = BME680Connector({"location": "Indoor", "gas_baseline": 50000.0})
        connector._sensor = MagicMock()

        mock_data = MagicMock()
        mock_data.temperature = 28.5
        mock_data.humidity = 65.0
        mock_data.heat_stable = True
        mock_data.gas_resistance = 50000.0

        connector._sensor.get_sensor_data.return_value = True
        connector._sensor.data = mock_data

        result = await connector.read()

        assert isinstance(result, AirQualityData)
        assert result.temperature == 28.5
        assert result.humidity == 65.0
        assert result.source == "bme680"
        assert result.location == "Indoor"
        assert result.aqi is not None

    @pytest.mark.asyncio
    async def test_read_returns_none_when_data_not_ready(self):
        connector = BME680Connector({})
        connector._sensor = MagicMock()
        connector._sensor.get_sensor_data.return_value = False

        result = await connector.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_aqi_none_when_heat_not_stable(self):
        connector = BME680Connector({"gas_baseline": 50000.0})
        connector._sensor = MagicMock()

        mock_data = MagicMock()
        mock_data.temperature = 25.0
        mock_data.humidity = 60.0
        mock_data.heat_stable = False

        connector._sensor.get_sensor_data.return_value = True
        connector._sensor.data = mock_data

        result = await connector.read()

        assert result is not None
        assert result.aqi is None
        assert result.temperature == 25.0

    def test_read_sensor_aqi_capped_at_500(self):
        """Test AQI does not exceed 500 even with very low gas resistance."""
        connector = BME680Connector({"gas_baseline": 50000.0})
        connector._sensor = MagicMock()

        mock_data = MagicMock()
        mock_data.temperature = 30.0
        mock_data.humidity = 70.0
        mock_data.heat_stable = True
        mock_data.gas_resistance = 1.0  # Extremely low — very polluted

        connector._sensor.get_sensor_data.return_value = True
        connector._sensor.data = mock_data

        result = connector._read_sensor()
        assert result is not None
        assert result.aqi <= 500

    def test_read_sensor_aqi_zero_floor(self):
        """Test AQI does not go below 0 with very high gas resistance."""
        connector = BME680Connector({"gas_baseline": 50000.0})
        connector._sensor = MagicMock()

        mock_data = MagicMock()
        mock_data.temperature = 22.0
        mock_data.humidity = 50.0
        mock_data.heat_stable = True
        mock_data.gas_resistance = 999999.0  # Extremely clean air

        connector._sensor.get_sensor_data.return_value = True
        connector._sensor.data = mock_data

        result = connector._read_sensor()
        assert result is not None
        assert result.aqi >= 0


class TestBME680ConnectorReadException:
    """Cover except Exception in read()."""

    @pytest.mark.asyncio
    async def test_read_executor_raises_exception(self):
        connector = BME680Connector({})
        connector._sensor = MagicMock()
        with patch.object(
            connector, "_read_sensor", side_effect=Exception("executor error")
        ):
            result = await connector.read()
        assert result is None
