"""
Weather Provider for OM1.

This module provides weather information to OM1 agents,
allowing robots to make weather-aware decisions.
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

from .singleton import singleton


@singleton
class WeatherProvider:
    """
    Singleton provider for fetching current weather data.

    This provider allows OM1 agents to access real-time weather
    information for location-aware and weather-aware behaviors.

    Uses wttr.in API (no API key required) by default.
    """

    DEFAULT_LOCATION = "auto"
    DEFAULT_CACHE_TTL = 300
    DEFAULT_TIMEOUT = 10

    def __init__(
        self,
        location: str = DEFAULT_LOCATION,
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ):
        """
        Initialize the WeatherProvider.

        Parameters
        ----------
        location : str, optional
            Location for weather data. Can be city name, coordinates,
            or "auto" for IP-based detection. Default is "auto".
        cache_ttl : int, optional
            Cache time-to-live in seconds. Default is 300 (5 minutes).
        """
        self.location = location
        self.cache_ttl = cache_ttl

        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._running = False

        logging.info(f"WeatherProvider initialized for location: {location}")

    def start(self) -> None:
        """
        Start the WeatherProvider.
        """
        self._running = True
        logging.info("WeatherProvider started")

    def stop(self) -> None:
        """
        Stop the WeatherProvider and clean up resources.
        """
        self._running = False
        self._cache = None
        self._cache_time = 0
        logging.info("WeatherProvider stopped")

    def _is_cache_valid(self) -> bool:
        """
        Check if cached weather data is still valid.

        Returns
        -------
        bool
            True if cache is valid, False otherwise.
        """
        if self._cache is None:
            return False
        return (time.time() - self._cache_time) < self.cache_ttl

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """
        Safely convert a value to int.

        Parameters
        ----------
        value : Any
            Value to convert.
        default : int
            Default value if conversion fails.

        Returns
        -------
        int
            Converted integer or default.
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float.

        Parameters
        ----------
        value : Any
            Value to convert.
        default : float
            Default value if conversion fails.

        Returns
        -------
        float
            Converted float or default.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _fetch_weather(self) -> Dict[str, Any]:
        """
        Fetch weather data from wttr.in API.

        Returns
        -------
        Dict[str, Any]
            Weather data dictionary.

        Raises
        ------
        requests.RequestException
            If the API request fails.
        """
        location_param = "" if self.location == "auto" else self.location
        url = f"https://wttr.in/{location_param}?format=j1"

        response = requests.get(url, timeout=self.DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        current = data.get("current_condition", [{}])[0]
        nearest_area = data.get("nearest_area", [{}])[0]

        area_name = nearest_area.get("areaName", [{}])
        country_name = nearest_area.get("country", [{}])
        weather_desc = current.get("weatherDesc", [{}])

        return {
            "location": (
                area_name[0].get("value", "Unknown") if area_name else "Unknown"
            ),
            "country": (
                country_name[0].get("value", "Unknown") if country_name else "Unknown"
            ),
            "temperature_c": self._safe_int(current.get("temp_C", 0)),
            "temperature_f": self._safe_int(current.get("temp_F", 32)),
            "feels_like_c": self._safe_int(current.get("FeelsLikeC", 0)),
            "feels_like_f": self._safe_int(current.get("FeelsLikeF", 32)),
            "humidity": self._safe_int(current.get("humidity", 0)),
            "condition": (
                weather_desc[0].get("value", "Unknown") if weather_desc else "Unknown"
            ),
            "wind_speed_kmh": self._safe_int(current.get("windspeedKmph", 0)),
            "wind_speed_mph": self._safe_int(current.get("windspeedMiles", 0)),
            "wind_direction": current.get("winddir16Point", "N"),
            "visibility_km": self._safe_int(current.get("visibility", 10)),
            "uv_index": self._safe_int(current.get("uvIndex", 0)),
            "precipitation_mm": self._safe_float(current.get("precipMM", 0)),
            "cloud_cover": self._safe_int(current.get("cloudcover", 0)),
            "timestamp": time.time(),
            "source": "wttr.in",
        }

    def get_weather(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current weather data.

        Parameters
        ----------
        force_refresh : bool, optional
            If True, bypass cache and fetch fresh data. Default is False.

        Returns
        -------
        Dict[str, Any]
            Weather data dictionary containing temperature, humidity,
            condition, wind, and other weather information.

        Raises
        ------
        Exception
            If weather data cannot be fetched.
        """
        if not force_refresh and self._is_cache_valid():
            logging.debug("Returning cached weather data")
            return self._cache

        try:
            weather_data = self._fetch_weather()
            self._cache = weather_data
            self._cache_time = time.time()

            logging.info(
                f"Weather updated: {weather_data['condition']} "
                f"{weather_data['temperature_c']}C in "
                f"{weather_data['location']}"
            )

            return weather_data

        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            if self._cache:
                logging.warning("Returning stale cached weather data")
                return self._cache
            raise

    def get_weather_summary(self) -> str:
        """
        Get a human-readable weather summary.

        Returns
        -------
        str
            Natural language weather summary for robot speech.
        """
        try:
            weather = self.get_weather()
            return (
                f"It's currently {weather['condition'].lower()} "
                f"and {weather['temperature_f']} degrees Fahrenheit "
                f"in {weather['location']}. "
                f"Humidity is {weather['humidity']} percent."
            )
        except Exception as e:
            logging.error(f"Error generating weather summary: {e}")
            return "I'm unable to get the current weather information."

    def is_good_weather(self) -> bool:
        """
        Determine if current weather is suitable for outdoor activities.

        Returns
        -------
        bool
            True if weather is good, False otherwise.
        """
        try:
            weather = self.get_weather()
            bad_conditions = [
                "rain",
                "storm",
                "snow",
                "sleet",
                "hail",
                "thunder",
                "blizzard",
                "hurricane",
                "tornado",
            ]
            condition_lower = weather["condition"].lower()

            for bad in bad_conditions:
                if bad in condition_lower:
                    return False

            if weather["temperature_c"] < 5 or weather["temperature_c"] > 35:
                return False

            if weather["precipitation_mm"] > 1:
                return False

            return True

        except Exception:
            return True

    @property
    def current_temperature_c(self) -> int:
        """Get current temperature in Celsius."""
        return self.get_weather().get("temperature_c", 0)

    @property
    def current_temperature_f(self) -> int:
        """Get current temperature in Fahrenheit."""
        return self.get_weather().get("temperature_f", 32)

    @property
    def current_condition(self) -> str:
        """Get current weather condition."""
        return self.get_weather().get("condition", "Unknown")
