"""Tests for Weather input plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.weather import Weather, WeatherConfig


class TestWeatherInitialization:
    """Tests for Weather input plugin initialization."""

    def test_initialization_default_config(self):
        """Test basic initialization with default config."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)

            assert sensor.messages == []
            assert sensor.descriptor_for_LLM == "Weather Information"
            assert sensor._first_poll is True
            assert sensor._consecutive_failures == 0
            mock_provider.start.assert_called_once()

    def test_initialization_custom_location(self):
        """Test initialization with custom location."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig(location="London")
            Weather(config=config)

            mock_provider_class.assert_called_once_with(location="London")


class TestWeatherPolling:
    """Tests for Weather input plugin polling."""

    @pytest.mark.asyncio
    async def test_poll_first_poll_no_sleep(self):
        """Test that first poll happens immediately without sleep."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
            patch(
                "inputs.plugins.weather.asyncio.sleep", new=AsyncMock()
            ) as mock_sleep,
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.return_value = {"condition": "Sunny"}
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)

            result = await sensor._poll()

            mock_sleep.assert_not_called()
            assert result == {"condition": "Sunny"}
            assert sensor._first_poll is False

    @pytest.mark.asyncio
    async def test_poll_subsequent_poll_sleeps(self):
        """Test that subsequent polls wait 60 seconds."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
            patch(
                "inputs.plugins.weather.asyncio.sleep", new=AsyncMock()
            ) as mock_sleep,
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.return_value = {"condition": "Cloudy"}
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)
            sensor._first_poll = False  # Simulate not first poll

            result = await sensor._poll()

            mock_sleep.assert_called_once_with(60)
            assert result == {"condition": "Cloudy"}

    @pytest.mark.asyncio
    async def test_poll_with_valid_data(self):
        """Test _poll when weather data is available."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.return_value = {
                "condition": "Sunny",
                "temperature_c": 25,
                "humidity": 50,
            }
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)

            result = await sensor._poll()

            assert result is not None
            assert result["condition"] == "Sunny"
            assert result["temperature_c"] == 25
            assert sensor._consecutive_failures == 0


class TestWeatherErrorHandling:
    """Tests for Weather input plugin error handling."""

    @pytest.mark.asyncio
    async def test_poll_error_increments_failure_count(self):
        """Test that polling errors increment failure counter."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.side_effect = Exception("API Error")
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)

            result = await sensor._poll()

            assert result is None
            assert sensor._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_poll_error_logs_warning_after_three_failures(self):
        """Test that warning is logged after 3 consecutive failures."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
            patch("inputs.plugins.weather.logging") as mock_logging,
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.side_effect = Exception("API Error")
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)
            sensor._consecutive_failures = 2  # Already 2 failures

            await sensor._poll()

            assert sensor._consecutive_failures == 3
            mock_logging.warning.assert_called()

    @pytest.mark.asyncio
    async def test_poll_success_resets_failure_count(self):
        """Test that successful poll resets failure counter."""
        with (
            patch("inputs.plugins.weather.WeatherProvider") as mock_provider_class,
            patch("inputs.plugins.weather.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.get_weather.return_value = {"condition": "Sunny"}
            mock_provider_class.return_value = mock_provider

            config = WeatherConfig()
            sensor = Weather(config=config)
            sensor._consecutive_failures = 2  # Had failures before

            await sensor._poll()

            assert sensor._consecutive_failures == 0


class TestWeatherRawToText:
    """Tests for Weather input plugin raw_to_text conversion."""

    @pytest.mark.asyncio
    async def test_raw_to_text_with_valid_data(self):
        """Test _raw_to_text with valid weather data."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
            patch("inputs.plugins.weather.time.time", return_value=1234.0),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)

            weather_data = {
                "condition": "Sunny",
                "temperature_c": 25,
                "feels_like_c": 27,
                "humidity": 50,
                "wind_speed_kmh": 10,
                "wind_direction": "N",
                "location": "TestCity",
                "country": "TestCountry",
                "uv_index": 5,
                "visibility_km": 10,
            }

            result = await sensor._raw_to_text(weather_data)

            assert result is not None
            assert result.timestamp == 1234.0
            assert "Sunny" in result.message
            assert "25C" in result.message
            assert "TestCity" in result.message
            assert "50%" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none(self):
        """Test _raw_to_text with None input."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)

            result = await sensor._raw_to_text(None)

            assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_updates_messages(self):
        """Test raw_to_text adds message to buffer."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
            patch("inputs.plugins.weather.time.time", return_value=1234.0),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)

            weather_data = {
                "condition": "Cloudy",
                "temperature_c": 18,
                "feels_like_c": 17,
                "humidity": 70,
                "wind_speed_kmh": 15,
                "wind_direction": "E",
                "location": "Paris",
                "country": "France",
                "uv_index": 2,
                "visibility_km": 8,
            }

            await sensor.raw_to_text(weather_data)

            assert len(sensor.messages) == 1
            assert "Cloudy" in sensor.messages[0].message


class TestWeatherBufferFormatting:
    """Tests for Weather input plugin buffer formatting."""

    def test_formatted_latest_buffer_with_messages(self):
        """Test formatted_latest_buffer with messages."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)
            sensor.io_provider = MagicMock()

            sensor.messages = [
                Message(timestamp=1000.0, message="Weather: Sunny, 25C"),
            ]

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Weather Information" in result
            assert "Sunny" in result
            sensor.io_provider.add_input.assert_called_once()
            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_empty(self):
        """Test formatted_latest_buffer with empty buffer."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)

            result = sensor.formatted_latest_buffer()

            assert result is None

    def test_formatted_latest_buffer_uses_latest_message(self):
        """Test formatted_latest_buffer uses only the latest message."""
        with (
            patch("inputs.plugins.weather.WeatherProvider"),
            patch("inputs.plugins.weather.IOProvider"),
        ):
            config = WeatherConfig()
            sensor = Weather(config=config)
            sensor.io_provider = MagicMock()

            sensor.messages = [
                Message(timestamp=1000.0, message="Weather: Rainy"),
                Message(timestamp=2000.0, message="Weather: Sunny"),
            ]

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Sunny" in result
            assert len(sensor.messages) == 0
