import asyncio
import logging
import time
from typing import Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class WeatherConfig(SensorConfig):
    """
    Configuration for Weather Input.

    Parameters
    ----------
    api_key : str
        OpenWeatherMap API key.
    latitude : float
        Location latitude.
    longitude : float
        Location longitude.
    poll_interval : float
        Seconds between weather updates (default: 300 = 5 minutes).
    units : str
        Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), 'standard' (Kelvin).
    """

    api_key: str = Field(description="OpenWeatherMap API key")
    latitude: float = Field(default=40.7128, description="Location latitude")
    longitude: float = Field(default=-74.0060, description="Location longitude")
    poll_interval: float = Field(
        default=300.0, description="Seconds between weather updates"
    )
    units: str = Field(default="metric", description="Temperature units")


class WeatherInput(FuserInput[WeatherConfig, Optional[dict]]):
    """
    Weather input that fetches current weather data from OpenWeatherMap API.

    Provides real-time weather information including temperature, conditions,
    humidity, and wind speed to give the robot environmental awareness.
    """

    def __init__(self, config: WeatherConfig):
        """
        Initialize the Weather input.

        Parameters
        ----------
        config : WeatherConfig
            Configuration for the weather sensor.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Current Weather"

        self.api_key = config.api_key
        self.latitude = config.latitude
        self.longitude = config.longitude
        self.poll_interval = config.poll_interval
        self.units = config.units

        self._last_poll_time: float = 0

        if not self.api_key:
            logging.warning("WeatherInput: API key not provided")

    async def _fetch_weather(self) -> Optional[dict]:
        """
        Fetch weather data from OpenWeatherMap API.

        Returns
        -------
        Optional[dict]
            Weather data or None if request failed.
        """
        if not self.api_key:
            return None

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
            "appid": self.api_key,
            "units": self.units,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logging.debug(f"WeatherInput: Fetched weather data: {data}")
                        return data
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"WeatherInput: API error {response.status}: {error_text}"
                        )
                        return None
        except asyncio.TimeoutError:
            logging.error("WeatherInput: Request timed out")
            return None
        except aiohttp.ClientError as e:
            logging.error(f"WeatherInput: Network error: {e}")
            return None
        except Exception as e:
            logging.error(f"WeatherInput: Unexpected error: {e}")
            return None

    async def _poll(self) -> Optional[dict]:
        """
        Poll for weather data based on poll_interval.

        Returns
        -------
        Optional[dict]
            Fresh weather data when poll interval has elapsed, None otherwise.
        """
        current_time = time.time()

        if current_time - self._last_poll_time < self.poll_interval:
            await asyncio.sleep(1.0)
            return None

        self._last_poll_time = current_time
        await asyncio.sleep(1.0)
        return await self._fetch_weather()

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Convert raw weather data to human-readable text.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw weather data from API.

        Returns
        -------
        Optional[Message]
            Formatted weather message or None.
        """
        if raw_input is None:
            return None

        try:
            main = raw_input.get("main", {})
            weather = raw_input.get("weather", [{}])[0]
            wind = raw_input.get("wind", {})
            location = raw_input.get("name", "Unknown location")

            temp = main.get("temp")
            feels_like = main.get("feels_like")
            humidity = main.get("humidity")
            condition = weather.get("description", "unknown")
            wind_speed = wind.get("speed")

            if self.units == "metric":
                unit_symbol = "°C"
                wind_unit = "m/s"
            elif self.units == "imperial":
                unit_symbol = "°F"
                wind_unit = "mph"
            else:
                unit_symbol = " K"
                wind_unit = "m/s"

            parts = [f"Weather in {location}: {condition}"]

            if temp is not None:
                parts.append(f"Temperature: {temp}{unit_symbol}")
            if feels_like is not None:
                parts.append(f"feels like {feels_like}{unit_symbol}")
            if humidity is not None:
                parts.append(f"Humidity: {humidity}%")
            if wind_speed is not None:
                parts.append(f"Wind: {wind_speed} {wind_unit}")

            message = ". ".join(parts) + "."
            return Message(timestamp=time.time(), message=message)

        except Exception as e:
            logging.error(f"WeatherInput: Error parsing weather data: {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer with processed weather data.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw weather data to be processed.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string for LLM or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
