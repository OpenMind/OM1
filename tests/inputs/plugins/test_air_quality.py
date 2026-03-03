import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.air_quality import AirQualityConfig, AirQualityInput
from inputs.plugins.air_quality.connector.base import AirQualityData


class TestAirQualityConfig:
    """Tests for AirQualityConfig."""

    def test_default_values(self):
        config = AirQualityConfig()
        assert config.connector == "aqicn"
        assert config.connector_config == {}
        assert config.poll_interval == 300.0
        assert config.aqi_warning_threshold == 100
        assert config.aqi_danger_threshold == 150

    def test_custom_values(self):
        config = AirQualityConfig(
            connector="pms5003",
            connector_config={"port": "/dev/ttyUSB0"},
            poll_interval=60.0,
            aqi_warning_threshold=75,
            aqi_danger_threshold=125,
        )
        assert config.connector == "pms5003"
        assert config.connector_config == {"port": "/dev/ttyUSB0"}
        assert config.poll_interval == 60.0
        assert config.aqi_warning_threshold == 75
        assert config.aqi_danger_threshold == 125


class TestAirQualityInputInit:
    """Tests for AirQualityInput initialization."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.air_quality.IOProvider") as mock:
            yield mock

    def test_init_default_connector(self, mock_io_provider):
        config = AirQualityConfig()
        plugin = AirQualityInput(config)
        assert plugin.descriptor_for_LLM == "Air Quality"
        assert plugin.poll_interval == 300.0
        assert plugin.aqi_warning_threshold == 100
        assert plugin.aqi_danger_threshold == 150
        assert plugin.messages == []

    def test_init_pms5003_connector(self, mock_io_provider):
        config = AirQualityConfig(
            connector="pms5003",
            connector_config={"port": "/dev/ttyUSB0"},
        )
        plugin = AirQualityInput(config)
        assert plugin._connector is not None

    def test_init_bme680_connector(self, mock_io_provider):
        config = AirQualityConfig(connector="bme680")
        plugin = AirQualityInput(config)
        assert plugin._connector is not None

    def test_init_unknown_connector_raises(self, mock_io_provider):
        config = AirQualityConfig(connector="unknown_sensor")
        with pytest.raises(ValueError, match="unknown connector"):
            AirQualityInput(config)


class TestAirQualityInputPoll:
    """Tests for _poll() behavior."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.air_quality.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def plugin(self, mock_io_provider):
        config = AirQualityConfig(poll_interval=60.0)
        return AirQualityInput(config)

    @pytest.mark.asyncio
    async def test_poll_returns_data_on_first_call(self, plugin):
        mock_data = AirQualityData(aqi=75, pm25=18.0, location="Test", source="aqicn")

        with (
            patch.object(
                plugin._connector, "connect", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                plugin._connector,
                "read",
                new_callable=AsyncMock,
                return_value=mock_data,
            ),
            patch.object(plugin._connector, "disconnect", new_callable=AsyncMock),
        ):
            result = await plugin._poll()

        assert result == mock_data

    @pytest.mark.asyncio
    async def test_poll_returns_none_before_interval(self, plugin):
        mock_data = AirQualityData(aqi=75, location="Test", source="aqicn")

        with (
            patch.object(
                plugin._connector, "connect", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                plugin._connector,
                "read",
                new_callable=AsyncMock,
                return_value=mock_data,
            ),
            patch.object(plugin._connector, "disconnect", new_callable=AsyncMock),
        ):
            await plugin._poll()
            result = await plugin._poll()

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_fetches_again_after_interval(self, plugin):
        mock_data = AirQualityData(aqi=75, location="Test", source="aqicn")

        with (
            patch.object(
                plugin._connector, "connect", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                plugin._connector,
                "read",
                new_callable=AsyncMock,
                return_value=mock_data,
            ),
            patch.object(plugin._connector, "disconnect", new_callable=AsyncMock),
        ):
            await plugin._poll()
            plugin._last_poll_time = time.time() - 120.0
            result = await plugin._poll()

        assert result == mock_data

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_connect_fails(self, plugin):
        with patch.object(
            plugin._connector, "connect", new_callable=AsyncMock, return_value=False
        ):
            result = await plugin._poll()
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_disconnects_after_read(self, plugin):
        mock_data = AirQualityData(aqi=50, location="Test", source="aqicn")

        with (
            patch.object(
                plugin._connector, "connect", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                plugin._connector,
                "read",
                new_callable=AsyncMock,
                return_value=mock_data,
            ) as mock_read,
            patch.object(
                plugin._connector, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            await plugin._poll()

        mock_read.assert_called_once()
        mock_disconnect.assert_called_once()


class TestAirQualityInputRawToText:
    """Tests for _raw_to_text conversion."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.air_quality.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def plugin(self, mock_io_provider):
        config = AirQualityConfig()
        return AirQualityInput(config)

    @pytest.mark.asyncio
    async def test_raw_to_text_none_returns_none(self, plugin):
        result = await plugin._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_full_data(self, plugin):
        data = AirQualityData(
            aqi=78,
            pm25=22.5,
            pm10=45.0,
            temperature=31.0,
            humidity=80.0,
            location="Semarang",
            source="aqicn",
        )
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "Semarang" in result.message
        assert "78" in result.message
        assert "PM2.5" in result.message
        assert "PM10" in result.message
        assert "31.0" in result.message
        assert "80.0" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_warning_threshold(self, plugin):
        data = AirQualityData(aqi=110, location="Test", source="aqicn")
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "WARNING" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_danger_threshold(self, plugin):
        data = AirQualityData(aqi=175, location="Test", source="aqicn")
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "DANGER" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_good_aqi_no_alert(self, plugin):
        data = AirQualityData(aqi=40, location="Test", source="aqicn")
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "WARNING" not in result.message
        assert "DANGER" not in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_no_aqi(self, plugin):
        """Test that missing AQI is handled gracefully."""
        data = AirQualityData(
            temperature=28.0,
            humidity=65.0,
            location="Indoor",
            source="bme680",
        )
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "Indoor" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_partial_data(self, plugin):
        """Test that missing optional fields are skipped."""
        data = AirQualityData(aqi=55, location="Test", source="pms5003")
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "µg/m³" not in result.message  # no pollutant data


class TestAirQualityInputFormatted:
    """Tests for formatted_latest_buffer."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.air_quality.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def plugin(self, mock_io_provider):
        config = AirQualityConfig()
        plugin = AirQualityInput(config)
        plugin.io_provider = mock_io_provider
        return plugin

    def test_formatted_empty_buffer_returns_none(self, plugin):
        result = plugin.formatted_latest_buffer()
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_with_message(self, plugin):
        data = AirQualityData(aqi=78, location="Semarang", source="aqicn")
        await plugin.raw_to_text(data)

        result = plugin.formatted_latest_buffer()
        assert result is not None
        assert "Air Quality" in result
        assert "// START" in result
        assert "// END" in result
        assert "Semarang" in result

    @pytest.mark.asyncio
    async def test_formatted_clears_buffer(self, plugin):
        data = AirQualityData(aqi=78, location="Test", source="aqicn")
        await plugin.raw_to_text(data)
        assert len(plugin.messages) == 1

        plugin.formatted_latest_buffer()
        assert len(plugin.messages) == 0

    @pytest.mark.asyncio
    async def test_formatted_calls_io_provider(self, plugin):
        data = AirQualityData(aqi=78, location="Test", source="aqicn")
        await plugin.raw_to_text(data)
        plugin.formatted_latest_buffer()
        plugin.io_provider.add_input.assert_called_once()


class TestAirQualityInputMissingCoverage:
    """Additional tests to reach 100% coverage."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.air_quality.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def plugin(self, mock_io_provider):
        config = AirQualityConfig()
        return AirQualityInput(config)

    @pytest.mark.asyncio
    async def test_raw_to_text_with_so2_no2_co_o3(self, plugin):
        """Cover so2, no2, co, o3 pollutant branches."""
        data = AirQualityData(
            aqi=55,
            co=0.8,
            no2=15.0,
            so2=5.0,
            o3=60.0,
            location="Test",
            source="aqicn",
        )
        result = await plugin._raw_to_text(data)
        assert result is not None
        assert "CO" in result.message
        assert "NO2" in result.message
        assert "SO2" in result.message
        assert "O3" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_none_does_not_append(self, plugin):
        """Cover raw_to_text when _raw_to_text returns None (pending is None)."""
        with patch.object(
            plugin, "_raw_to_text", new_callable=AsyncMock, return_value=None
        ):
            await plugin.raw_to_text(AirQualityData(location="Test", source="x"))
        assert len(plugin.messages) == 0

    @pytest.mark.asyncio
    async def test_raw_to_text_exception_returns_none(self, plugin):
        """Cover except Exception in _raw_to_text."""
        data = MagicMock()
        data.aqi = "not_a_number"  # will cause comparison error
        data.location = "Test"
        data.source = "x"
        data.pm25 = None
        data.pm10 = None
        data.co = None
        data.no2 = None
        data.so2 = None
        data.o3 = None
        data.temperature = None
        data.humidity = None
        result = await plugin._raw_to_text(data)
        assert result is None
