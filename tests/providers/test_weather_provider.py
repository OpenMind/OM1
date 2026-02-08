"""
Tests for weather_provider.

This module contains tests for the WeatherProvider class.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock dependencies before import
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()


def mock_singleton(cls):
    """Mock singleton decorator."""
    return cls


sys.modules["providers.singleton"] = MagicMock()
sys.modules["providers.singleton"].singleton = mock_singleton


class TestWeatherProviderInitialization:
    """Tests for WeatherProvider initialization."""

    def test_initialization_default_location(self):
        """Test WeatherProvider initializes with default location."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider.location == "auto"

    def test_initialization_custom_location(self):
        """Test WeatherProvider initializes with custom location."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider(location="London")

        assert provider.location == "London"

    def test_initialization_default_cache_ttl(self):
        """Test WeatherProvider initializes with default cache TTL."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider.cache_ttl == 300

    def test_initialization_custom_cache_ttl(self):
        """Test WeatherProvider initializes with custom cache TTL."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider(cache_ttl=600)

        assert provider.cache_ttl == 600

    def test_initialization_running_is_false(self):
        """Test WeatherProvider running is False initially."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._running is False

    def test_initialization_cache_is_none(self):
        """Test WeatherProvider cache is None initially."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._cache is None


class TestWeatherProviderMethods:
    """Tests for WeatherProvider methods."""

    def test_start_sets_running_true(self):
        """Test start method sets running to True."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()
        provider.start()

        assert provider._running is True

    def test_stop_sets_running_false(self):
        """Test stop method sets running to False."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()
        provider.start()
        provider.stop()

        assert provider._running is False

    def test_stop_clears_cache(self):
        """Test stop method clears cache."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()
        provider._cache = {"test": "data"}
        provider.stop()

        assert provider._cache is None

    def test_is_cache_valid_returns_false_when_none(self):
        """Test _is_cache_valid returns False when cache is None."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._is_cache_valid() is False

    def test_safe_int_converts_valid_string(self):
        """Test _safe_int converts valid string."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._safe_int("42") == 42

    def test_safe_int_returns_default_on_invalid(self):
        """Test _safe_int returns default on invalid input."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._safe_int("invalid", 99) == 99

    def test_safe_float_converts_valid_string(self):
        """Test _safe_float converts valid string."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_returns_default_on_invalid(self):
        """Test _safe_float returns default on invalid input."""
        from providers.weather_provider import WeatherProvider

        provider = WeatherProvider()

        assert provider._safe_float("invalid", 1.5) == 1.5


class TestWeatherProviderWithMockedAPI:
    """Tests for WeatherProvider with mocked API responses."""

    @patch("providers.weather_provider.requests.get")
    def test_get_weather_returns_data(self, mock_get):
        """Test get_weather returns weather data."""
        from providers.weather_provider import WeatherProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "temp_C": "20",
                    "temp_F": "68",
                    "FeelsLikeC": "19",
                    "FeelsLikeF": "66",
                    "humidity": "65",
                    "weatherDesc": [{"value": "Sunny"}],
                    "windspeedKmph": "10",
                    "windspeedMiles": "6",
                    "winddir16Point": "N",
                    "visibility": "10",
                    "uvIndex": "3",
                    "precipMM": "0",
                    "cloudcover": "25",
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "Lagos"}],
                    "country": [{"value": "Nigeria"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        provider = WeatherProvider()
        weather = provider.get_weather()

        assert weather["temperature_c"] == 20
        assert weather["temperature_f"] == 68
        assert weather["condition"] == "Sunny"
        assert weather["location"] == "Lagos"

    @patch("providers.weather_provider.requests.get")
    def test_get_weather_summary_returns_string(self, mock_get):
        """Test get_weather_summary returns formatted string."""
        from providers.weather_provider import WeatherProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "temp_C": "25",
                    "temp_F": "77",
                    "FeelsLikeC": "24",
                    "FeelsLikeF": "75",
                    "humidity": "70",
                    "weatherDesc": [{"value": "Partly Cloudy"}],
                    "windspeedKmph": "15",
                    "windspeedMiles": "9",
                    "winddir16Point": "SW",
                    "visibility": "10",
                    "uvIndex": "5",
                    "precipMM": "0",
                    "cloudcover": "40",
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "TestCity"}],
                    "country": [{"value": "TestCountry"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        provider = WeatherProvider()
        summary = provider.get_weather_summary()

        assert "partly cloudy" in summary.lower()
        assert "77" in summary
        assert "TestCity" in summary


class TestWeatherProviderProperties:
    """Tests for WeatherProvider properties."""

    @patch("providers.weather_provider.requests.get")
    def test_current_temperature_c_property(self, mock_get):
        """Test current_temperature_c property returns correct value."""
        from providers.weather_provider import WeatherProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "temp_C": "30",
                    "temp_F": "86",
                    "FeelsLikeC": "32",
                    "FeelsLikeF": "90",
                    "humidity": "60",
                    "weatherDesc": [{"value": "Clear"}],
                    "windspeedKmph": "5",
                    "windspeedMiles": "3",
                    "winddir16Point": "N",
                    "visibility": "10",
                    "uvIndex": "8",
                    "precipMM": "0",
                    "cloudcover": "0",
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "TestCity"}],
                    "country": [{"value": "TestCountry"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        provider = WeatherProvider()

        assert provider.current_temperature_c == 30

    @patch("providers.weather_provider.requests.get")
    def test_current_condition_property(self, mock_get):
        """Test current_condition property returns correct value."""
        from providers.weather_provider import WeatherProvider

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "temp_C": "18",
                    "temp_F": "64",
                    "FeelsLikeC": "17",
                    "FeelsLikeF": "63",
                    "humidity": "75",
                    "weatherDesc": [{"value": "Overcast"}],
                    "windspeedKmph": "8",
                    "windspeedMiles": "5",
                    "winddir16Point": "W",
                    "visibility": "10",
                    "uvIndex": "2",
                    "precipMM": "0",
                    "cloudcover": "90",
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "TestCity"}],
                    "country": [{"value": "TestCountry"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        provider = WeatherProvider()

        assert provider.current_condition == "Overcast"
