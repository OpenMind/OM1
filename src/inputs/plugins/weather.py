import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.weather_provider import WeatherProvider


class WeatherConfig(SensorConfig):
    """
    Configuration for the Weather input plugin.

    Parameters
    ----------
    location : str
        Location for weather data. Can be city name, coordinates,
        or "auto" for IP-based detection. Default is "auto".
    """

    location: str = "auto"


class Weather(FuserInput[WeatherConfig, Optional[dict]]):
    """
    Weather input handler for reading weather data.

    Polls weather information periodically and provides it to
    the LLM as context for weather-aware responses.
    """

    def __init__(self, config: WeatherConfig):
        """
        Initialize the Weather input handler.

        Parameters
        ----------
        config : WeatherConfig
            Configuration for the weather sensor.
        """
        super().__init__(config)

        self.weather_provider = WeatherProvider(location=config.location)
        self.weather_provider.start()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Weather Information"
        self._first_poll = True
        self._consecutive_failures = 0

    async def _poll(self) -> Optional[dict]:
        """
        Poll for weather data from the WeatherProvider.

        First poll happens immediately on startup, subsequent polls
        wait 60 seconds since weather data does not change frequently.

        Returns
        -------
        Optional[dict]
            Weather data dictionary if available, None otherwise.
        """
        if self._first_poll:
            self._first_poll = False
        else:
            await asyncio.sleep(60)

        try:
            data = self.weather_provider.get_weather()
            self._consecutive_failures = 0
            return data
        except Exception as e:
            self._consecutive_failures += 1
            logging.error(
                f"Weather polling error (attempt {self._consecutive_failures}): {e}"
            )
            if self._consecutive_failures >= 3:
                logging.warning(
                    "Weather API unreachable after 3 attempts. "
                    "Will keep retrying every 60 seconds."
                )
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Convert raw weather data to a human-readable message.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw weather data dictionary from WeatherProvider.

        Returns
        -------
        Optional[Message]
            A timestamped message with formatted weather info, or None.
        """
        if not raw_input:
            return None

        logging.debug(f"Weather data: {raw_input}")

        condition = raw_input.get("condition", "Unknown")
        temp_c = raw_input.get("temperature_c", 0)
        feels_like_c = raw_input.get("feels_like_c", 0)
        humidity = raw_input.get("humidity", 0)
        wind_speed = raw_input.get("wind_speed_kmh", 0)
        wind_dir = raw_input.get("wind_direction", "N")
        location = raw_input.get("location", "Unknown")
        country = raw_input.get("country", "Unknown")
        uv_index = raw_input.get("uv_index", 0)
        visibility = raw_input.get("visibility_km", 10)

        msg = (
            f"Current weather in {location}, {country}: "
            f"{condition}, {temp_c}C (feels like {feels_like_c}C). "
            f"Humidity: {humidity}%. "
            f"Wind: {wind_speed} km/h {wind_dir}. "
            f"UV index: {uv_index}. "
            f"Visibility: {visibility} km."
        )

        return Message(timestamp=time.time(), message=msg)

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

        Formats the most recent weather message with the descriptor,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
