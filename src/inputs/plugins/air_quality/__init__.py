import asyncio
import logging
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from inputs.plugins.air_quality.connector.aqicn import AqicnConnector
from inputs.plugins.air_quality.connector.base import (
    AirQualityConnector,
    AirQualityData,
    get_aqi_level,
)
from inputs.plugins.air_quality.connector.bme680 import BME680Connector
from inputs.plugins.air_quality.connector.pms5003 import PMS5003Connector
from providers.io_provider import IOProvider

CONNECTORS: dict[str, type[AirQualityConnector]] = {
    "aqicn": AqicnConnector,
    "pms5003": PMS5003Connector,
    "bme680": BME680Connector,
}


class AirQualityConfig(SensorConfig):
    """
    Configuration for AirQuality Input.

    Parameters
    ----------
    connector : str
        Connector type: 'aqicn', 'pms5003', or 'bme680'.
    connector_config : dict
        Connector-specific configuration passed directly to the connector.
        aqicn   → api_key, latitude, longitude
        pms5003 → port, location
        bme680  → i2c_address, location, gas_baseline
    poll_interval : float
        Seconds between air quality reads (default: 300).
    aqi_warning_threshold : int
        AQI above this value triggers a WARNING (default: 100).
    aqi_danger_threshold : int
        AQI above this value triggers a DANGER alert (default: 150).
    """

    connector: str = Field(
        default="aqicn",
        description="Connector type: 'aqicn', 'pms5003', 'bme680'",
    )
    connector_config: dict = Field(
        default_factory=dict,
        description="Connector-specific configuration",
    )
    poll_interval: float = Field(
        default=300.0,
        description="Seconds between air quality reads",
    )
    aqi_warning_threshold: int = Field(
        default=100,
        description="AQI threshold for WARNING alert",
    )
    aqi_danger_threshold: int = Field(
        default=150,
        description="AQI threshold for DANGER alert",
    )


class AirQualityInput(FuserInput[AirQualityConfig, Optional[AirQualityData]]):
    """
    Generic air quality input that works with any sensor or API connector.

    Reads standardized AirQualityData from the configured connector and
    converts it to human-readable text for the LLM. Supports hot-swapping
    connectors via config without changing this class.

    Supported connectors:
        aqicn   — AQICN cloud API (no hardware needed)
        pms5003 — PMS5003/PMS7003 particulate sensor via Serial
        bme680  — BME680 environmental sensor via I2C
    """

    def __init__(self, config: AirQualityConfig):
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Air Quality"

        self.poll_interval = config.poll_interval
        self.aqi_warning_threshold = config.aqi_warning_threshold
        self.aqi_danger_threshold = config.aqi_danger_threshold
        self._last_poll_time: float = 0

        connector_class = CONNECTORS.get(config.connector)
        if connector_class is None:
            raise ValueError(
                f"AirQualityInput: unknown connector '{config.connector}'. "
                f"Available: {list(CONNECTORS.keys())}"
            )
        self._connector: AirQualityConnector = connector_class(config.connector_config)
        logging.info(f"AirQualityInput: using connector '{config.connector}'")

    async def _poll(self) -> Optional[AirQualityData]:
        """
        Poll connector based on poll_interval.

        Returns
        -------
        Optional[AirQualityData]
            Fresh data when interval elapsed, None otherwise.
        """
        current_time = time.time()

        if current_time - self._last_poll_time < self.poll_interval:
            await asyncio.sleep(1.0)
            return None

        self._last_poll_time = current_time
        await asyncio.sleep(1.0)

        connected = await self._connector.connect()
        if not connected:
            return None

        data = await self._connector.read()
        await self._connector.disconnect()
        return data

    async def _raw_to_text(
        self, raw_input: Optional[AirQualityData]
    ) -> Optional[Message]:
        """
        Convert AirQualityData to human-readable message for LLM.

        Parameters
        ----------
        raw_input : Optional[AirQualityData]
            Standardized air quality data.

        Returns
        -------
        Optional[Message]
            Formatted message, or None if no data.
        """
        if raw_input is None:
            return None

        try:
            aqi = raw_input.aqi
            aqi_label: str = ""
            aqi_description: str = ""
            parts = []

            if aqi is not None:
                aqi_label, aqi_description = get_aqi_level(aqi)
                parts.append(
                    f"Air Quality in {raw_input.location}: {aqi_label} (AQI: {aqi})"
                )
            else:
                parts.append(f"Air Quality in {raw_input.location}")

            pollutants = []
            if raw_input.pm25 is not None:
                pollutants.append(f"PM2.5: {raw_input.pm25} µg/m³")
            if raw_input.pm10 is not None:
                pollutants.append(f"PM10: {raw_input.pm10} µg/m³")
            if raw_input.co is not None:
                pollutants.append(f"CO: {raw_input.co} ppm")
            if raw_input.no2 is not None:
                pollutants.append(f"NO2: {raw_input.no2} µg/m³")
            if raw_input.so2 is not None:
                pollutants.append(f"SO2: {raw_input.so2} µg/m³")
            if raw_input.o3 is not None:
                pollutants.append(f"O3: {raw_input.o3} µg/m³")
            if pollutants:
                parts.append(", ".join(pollutants))

            env_data = []
            if raw_input.temperature is not None:
                env_data.append(f"Temperature: {raw_input.temperature}°C")
            if raw_input.humidity is not None:
                env_data.append(f"Humidity: {raw_input.humidity}%")
            if env_data:
                parts.append(", ".join(env_data))

            if aqi is not None:
                if aqi >= self.aqi_danger_threshold:
                    parts.append(
                        f"DANGER: Air quality is {aqi_label} — {aqi_description}"
                    )
                elif aqi >= self.aqi_warning_threshold:
                    parts.append(
                        f"WARNING: Air quality is {aqi_label} — {aqi_description}"
                    )

            return Message(timestamp=time.time(), message=". ".join(parts) + ".")

        except Exception as e:
            logging.error(f"AirQualityInput: error building message: {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[AirQualityData]) -> None:
        """Convert raw AirQualityData to a human-readable text message."""
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return the latest formatted air quality message, or None if empty."""
        if not self.messages:
            return None

        latest = self.messages[-1]
        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )
        self.io_provider.add_input(
            self.descriptor_for_LLM, latest.message, latest.timestamp
        )
        self.messages = []
        return result
