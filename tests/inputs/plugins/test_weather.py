import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.weather import WeatherConfig, WeatherInput


class TestWeatherConfig:
    """Tests for WeatherConfig."""

    def test_default_values(self):
        """Test config with default values."""
        config = WeatherConfig(api_key="test_api_key")
        assert config.api_key == "test_api_key"
        assert config.latitude == 40.7128
        assert config.longitude == -74.0060
        assert config.poll_interval == 300.0
        assert config.units == "metric"

    def test_custom_values(self):
        """Test config with custom values."""
        config = WeatherConfig(
            api_key="my_key",
            latitude=51.5074,
            longitude=-0.1278,
            poll_interval=600.0,
            units="imperial",
        )
        assert config.latitude == 51.5074
        assert config.longitude == -0.1278
        assert config.poll_interval == 600.0
        assert config.units == "imperial"


class TestWeatherInput:
    """Tests for WeatherInput."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.weather.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def weather_input(self, mock_io_provider):
        """Create a WeatherInput instance."""
        config = WeatherConfig(
            api_key="test_api_key",
            latitude=40.7128,
            longitude=-74.0060,
            poll_interval=300.0,
        )
        return WeatherInput(config)

    def test_init_with_config(self, mock_io_provider):
        """Test initialization with config."""
        config = WeatherConfig(api_key="test_key", latitude=35.0, longitude=139.0)
        weather = WeatherInput(config)
        assert weather.api_key == "test_key"
        assert weather.latitude == 35.0
        assert weather.longitude == 139.0
        assert weather.descriptor_for_LLM == "Current Weather"

    def test_init_without_api_key_logs_warning(self, mock_io_provider):
        """Test that missing API key logs a warning."""
        with patch("inputs.plugins.weather.logging.warning") as mock_warning:
            config = WeatherConfig(api_key="")
            WeatherInput(config)
            mock_warning.assert_called_with("WeatherInput: API key not provided")


class TestWeatherInputFetch:
    """Tests for weather fetching."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.weather.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def weather_input(self, mock_io_provider):
        """Create a WeatherInput instance."""
        config = WeatherConfig(api_key="test_api_key")
        return WeatherInput(config)

    @pytest.mark.asyncio
    async def test_fetch_weather_success(self, weather_input):
        """Test successful weather fetch."""
        mock_response_data = {
            "name": "New York",
            "main": {"temp": 20.5, "feels_like": 19.0, "humidity": 65},
            "weather": [{"description": "partly cloudy"}],
            "wind": {"speed": 5.5},
        }

        with patch("inputs.plugins.weather.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await weather_input._fetch_weather()

            assert result is not None
            assert result["name"] == "New York"
            assert result["main"]["temp"] == 20.5

    @pytest.mark.asyncio
    async def test_fetch_weather_api_error(self, weather_input):
        """Test handling of API error."""
        with patch("inputs.plugins.weather.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Invalid API key")

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch("inputs.plugins.weather.logging.error") as mock_error:
                result = await weather_input._fetch_weather()
                assert result is None
                assert any("401" in str(call) for call in mock_error.call_args_list)

    @pytest.mark.asyncio
    async def test_fetch_weather_no_api_key(self, mock_io_provider):
        """Test that fetch returns None without API key."""
        config = WeatherConfig(api_key="")
        weather = WeatherInput(config)
        result = await weather._fetch_weather()
        assert result is None


class TestWeatherInputPoll:
    """Tests for _poll() behavior."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.weather.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def weather_input(self, mock_io_provider):
        """Create a WeatherInput instance with short poll interval."""
        config = WeatherConfig(api_key="test_api_key", poll_interval=60.0)
        return WeatherInput(config)

    @pytest.mark.asyncio
    async def test_poll_returns_data_on_first_call(self, weather_input):
        """Test that first poll fetches weather data."""
        mock_data = {"name": "NYC", "main": {"temp": 20.0}}

        with patch.object(
            weather_input, "_fetch_weather", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_data
            result = await weather_input._poll()

            assert result == mock_data
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_returns_none_before_interval(self, weather_input):
        """Test that poll returns None when interval has not elapsed."""
        mock_data = {"name": "NYC", "main": {"temp": 20.0}}

        with patch.object(
            weather_input, "_fetch_weather", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_data

            await weather_input._poll()
            mock_fetch.reset_mock()

            result = await weather_input._poll()
            assert result is None
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_fetches_again_after_interval(self, weather_input):
        """Test that poll fetches new data after interval elapses."""
        mock_data = {"name": "NYC", "main": {"temp": 20.0}}

        with patch.object(
            weather_input, "_fetch_weather", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_data

            await weather_input._poll()
            mock_fetch.reset_mock()

            weather_input._last_poll_time = time.time() - 120.0

            result = await weather_input._poll()
            assert result == mock_data
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_does_not_send_duplicate_data(self, weather_input):
        """Test that repeated polls don't create duplicate messages."""
        mock_data = {
            "name": "NYC",
            "main": {"temp": 20.0},
            "weather": [{"description": "clear"}],
        }

        with patch.object(
            weather_input, "_fetch_weather", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = mock_data

            result1 = await weather_input._poll()
            await weather_input.raw_to_text(result1)
            assert len(weather_input.messages) == 1

            result2 = await weather_input._poll()
            await weather_input.raw_to_text(result2)
            assert len(weather_input.messages) == 1


class TestWeatherInputRawToText:
    """Tests for raw_to_text conversion."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.weather.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def weather_input(self, mock_io_provider):
        """Create a WeatherInput instance."""
        config = WeatherConfig(api_key="test_api_key")
        return WeatherInput(config)

    @pytest.mark.asyncio
    async def test_raw_to_text_success(self, weather_input):
        """Test successful conversion to text."""
        raw_data = {
            "name": "London",
            "main": {"temp": 15.0, "feels_like": 14.0, "humidity": 80},
            "weather": [{"description": "light rain"}],
            "wind": {"speed": 3.0},
        }

        result = await weather_input._raw_to_text(raw_data)

        assert result is not None
        assert "London" in result.message
        assert "light rain" in result.message
        assert "15" in result.message
        assert "°C" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_imperial_units(self, mock_io_provider):
        """Test conversion with imperial units."""
        config = WeatherConfig(api_key="test_key", units="imperial")
        weather = WeatherInput(config)

        raw_data = {
            "name": "Miami",
            "main": {"temp": 85.0, "feels_like": 90.0, "humidity": 75},
            "weather": [{"description": "sunny"}],
            "wind": {"speed": 10.0},
        }

        result = await weather._raw_to_text(raw_data)

        assert result is not None
        assert "Miami" in result.message
        assert "°F" in result.message
        assert "mph" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, weather_input):
        """Test that None input returns None."""
        result = await weather_input._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_kelvin_units(self, mock_io_provider):
        """Test conversion with standard (Kelvin) units."""
        config = WeatherConfig(api_key="test_key", units="standard")
        weather = WeatherInput(config)

        raw_data = {
            "name": "Moscow",
            "main": {"temp": 263.15, "feels_like": 258.0, "humidity": 70},
            "weather": [{"description": "snow"}],
            "wind": {"speed": 7.0},
        }

        result = await weather._raw_to_text(raw_data)

        assert result is not None
        assert "Moscow" in result.message
        assert " K" in result.message
        assert "m/s" in result.message
        assert "°F" not in result.message
        assert "°C" not in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_partial_data(self, weather_input):
        """Test handling of partial data."""
        raw_data = {
            "name": "Tokyo",
            "main": {"temp": 25.0},
            "weather": [{"description": "clear sky"}],
        }

        result = await weather_input._raw_to_text(raw_data)

        assert result is not None
        assert "Tokyo" in result.message
        assert "clear sky" in result.message


class TestWeatherInputFormatted:
    """Tests for formatted_latest_buffer."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.weather.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def weather_input(self, mock_io_provider):
        """Create a WeatherInput instance."""
        config = WeatherConfig(api_key="test_api_key")
        weather = WeatherInput(config)
        weather.io_provider = mock_io_provider
        return weather

    def test_formatted_latest_buffer_empty(self, weather_input):
        """Test that empty buffer returns None."""
        result = weather_input.formatted_latest_buffer()
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_with_message(self, weather_input):
        """Test formatting with message in buffer."""
        raw_data = {
            "name": "Paris",
            "main": {"temp": 18.0, "humidity": 60},
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 4.0},
        }

        await weather_input.raw_to_text(raw_data)
        result = weather_input.formatted_latest_buffer()

        assert result is not None
        assert "Current Weather" in result
        assert "Paris" in result
        assert "// START" in result
        assert "// END" in result

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_clears_messages(self, weather_input):
        """Test that buffer is cleared after formatting."""
        raw_data = {
            "name": "Berlin",
            "main": {"temp": 10.0},
            "weather": [{"description": "foggy"}],
        }

        await weather_input.raw_to_text(raw_data)
        assert len(weather_input.messages) == 1

        weather_input.formatted_latest_buffer()
        assert len(weather_input.messages) == 0
