import pytest

from inputs.plugins.air_quality.connector.base import (
    AirQualityConnector,
    AirQualityData,
    get_aqi_level,
)


class TestAirQualityData:
    """Tests for AirQualityData dataclass."""

    def test_default_values(self):
        data = AirQualityData()
        assert data.aqi is None
        assert data.pm25 is None
        assert data.pm10 is None
        assert data.co is None
        assert data.no2 is None
        assert data.so2 is None
        assert data.o3 is None
        assert data.temperature is None
        assert data.humidity is None
        assert data.location == "Unknown"
        assert data.source == "Unknown"

    def test_custom_values(self):
        data = AirQualityData(
            aqi=75,
            pm25=15.2,
            pm10=30.5,
            temperature=28.0,
            humidity=65.0,
            location="Semarang",
            source="pms5003",
        )
        assert data.aqi == 75
        assert data.pm25 == 15.2
        assert data.pm10 == 30.5
        assert data.temperature == 28.0
        assert data.humidity == 65.0
        assert data.location == "Semarang"
        assert data.source == "pms5003"


class TestGetAqiLevel:
    """Tests for get_aqi_level function."""

    def test_good(self):
        label, description = get_aqi_level(25)
        assert label == "GOOD"

    def test_moderate(self):
        label, _ = get_aqi_level(75)
        assert label == "MODERATE"

    def test_unhealthy_sensitive(self):
        label, _ = get_aqi_level(125)
        assert label == "UNHEALTHY FOR SENSITIVE GROUPS"

    def test_unhealthy(self):
        label, _ = get_aqi_level(175)
        assert label == "UNHEALTHY"

    def test_very_unhealthy(self):
        label, _ = get_aqi_level(250)
        assert label == "VERY UNHEALTHY"

    def test_hazardous(self):
        label, _ = get_aqi_level(350)
        assert label == "HAZARDOUS"

    def test_boundary_good_moderate(self):
        label, _ = get_aqi_level(50)
        assert label == "GOOD"
        label, _ = get_aqi_level(51)
        assert label == "MODERATE"

    def test_description_not_empty(self):
        for aqi in [25, 75, 125, 175, 250, 350]:
            _, description = get_aqi_level(aqi)
            assert len(description) > 0


class TestAirQualityConnectorAbstract:
    """Tests that AirQualityConnector enforces abstract methods."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AirQualityConnector({})  # type: ignore[abstract]

    def test_concrete_must_implement_all_methods(self):
        class IncompleteConnector(AirQualityConnector):
            async def connect(self):
                return True

            # missing read() and disconnect()

        with pytest.raises(TypeError):
            IncompleteConnector({})  # type: ignore[abstract]

    def test_concrete_full_implementation(self):
        class ConcreteConnector(AirQualityConnector):
            async def connect(self):
                return True

            async def read(self):
                return None

            async def disconnect(self):
                pass

        connector = ConcreteConnector({})
        assert connector is not None
        assert connector.config == {}


class TestGetAqiLevelFallback:
    """Cover fallback return when AQI > 300."""

    def test_above_300_returns_hazardous(self):
        label, description = get_aqi_level(999)
        assert label == "HAZARDOUS"
        assert len(description) > 0
