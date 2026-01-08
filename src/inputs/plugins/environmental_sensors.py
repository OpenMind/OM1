import asyncio
import json
import logging
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.data_analytics_provider import DataAnalyticsProvider, SensorReading
from providers.io_provider import IOProvider


class EnvironmentalSensorsConfig(SensorConfig):
    """Configuration for Environmental Sensors Input."""

    temperature_topic: Optional[str] = Field(
        default=None, description="Zenoh topic for temperature sensor data"
    )
    humidity_topic: Optional[str] = Field(
        default=None, description="Zenoh topic for humidity sensor data"
    )
    air_quality_topic: Optional[str] = Field(
        default=None, description="Zenoh topic for air quality sensor data"
    )
    use_zenoh: bool = Field(
        default=False, description="Whether to use Zenoh for sensor data"
    )
    update_interval: float = Field(
        default=1.0, description="Interval in seconds between sensor readings"
    )


class EnvironmentalSensors(FuserInput[EnvironmentalSensorsConfig, Optional[dict]]):
    """Environmental Sensors Input Plugin."""

    def __init__(self, config: EnvironmentalSensorsConfig):
        super().__init__(config)

        self.io_provider = IOProvider()
        self.descriptor_for_LLM = "Environmental Sensors"
        self.messages: List[Message] = []
        self.temperature: Optional[float] = None
        self.humidity: Optional[float] = None
        self.air_quality: Optional[float] = None

        self.analytics_provider: Optional[DataAnalyticsProvider] = None
        try:
            self.analytics_provider = DataAnalyticsProvider()
            logging.info("Data analytics enabled for Environmental Sensors")
        except Exception as e:
            logging.warning(f"Could not initialize analytics provider: {e}")

        self.use_zenoh = self.config.use_zenoh
        self.session = None
        self.subscribers = []

        if self.use_zenoh:
            self._setup_zenoh()
        else:
            logging.info(
                "Environmental Sensors using mock data (use_zenoh=False). "
                "Set use_zenoh=True and provide topics to use real sensors."
            )

    def _setup_zenoh(self):
        try:
            from zenoh_msgs import open_zenoh_session

            self.session = open_zenoh_session()
            logging.info("Environmental Sensors Zenoh session initialized")

            if self.config.temperature_topic:
                subscriber = self.session.declare_subscriber(
                    self.config.temperature_topic, self._temperature_handler
                )
                self.subscribers.append(subscriber)
                logging.info(
                    f"Subscribed to temperature topic: {self.config.temperature_topic}"
                )

            if self.config.humidity_topic:
                subscriber = self.session.declare_subscriber(
                    self.config.humidity_topic, self._humidity_handler
                )
                self.subscribers.append(subscriber)
                logging.info(
                    f"Subscribed to humidity topic: {self.config.humidity_topic}"
                )

            if self.config.air_quality_topic:
                subscriber = self.session.declare_subscriber(
                    self.config.air_quality_topic, self._air_quality_handler
                )
                self.subscribers.append(subscriber)
                logging.info(
                    f"Subscribed to air quality topic: {self.config.air_quality_topic}"
                )

        except Exception as e:
            logging.error(f"Error setting up Zenoh for Environmental Sensors: {e}")
            self.use_zenoh = False

    def _temperature_handler(self, sample):
        try:
            data = json.loads(sample.payload.decode("utf-8"))
            if "temperature" in data:
                self.temperature = float(data["temperature"])
                logging.debug(f"Received temperature: {self.temperature}°C")
        except Exception as e:
            logging.error(f"Error processing temperature data: {e}")

    def _humidity_handler(self, sample):
        try:
            data = json.loads(sample.payload.decode("utf-8"))
            if "humidity" in data:
                self.humidity = float(data["humidity"])
                logging.debug(f"Received humidity: {self.humidity}%")
        except Exception as e:
            logging.error(f"Error processing humidity data: {e}")

    def _air_quality_handler(self, sample):
        try:
            data = json.loads(sample.payload.decode("utf-8"))
            if "air_quality" in data:
                self.air_quality = float(data["air_quality"])
                logging.debug(f"Received air quality: {self.air_quality}")
        except Exception as e:
            logging.error(f"Error processing air quality data: {e}")

    def _generate_mock_data(self) -> dict:
        import random

        self.temperature = round(20.0 + random.uniform(-5, 5), 2)
        self.humidity = round(50.0 + random.uniform(-20, 20), 2)
        self.air_quality = round(100.0 + random.uniform(-30, 30), 2)

        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "air_quality": self.air_quality,
            "timestamp": time.time(),
        }

    async def _poll(self) -> Optional[dict]:
        await asyncio.sleep(self.config.update_interval)

        if not self.use_zenoh:
            return self._generate_mock_data()
        if (
            self.temperature is not None
            or self.humidity is not None
            or self.air_quality is not None
        ):
            return {
                "temperature": self.temperature,
                "humidity": self.humidity,
                "air_quality": self.air_quality,
                "timestamp": time.time(),
            }

        return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        if raw_input is None:
            return None

        if self.analytics_provider:
            try:
                reading = SensorReading(
                    timestamp=raw_input.get("timestamp", time.time()),
                    temperature=raw_input.get("temperature"),
                    humidity=raw_input.get("humidity"),
                    air_quality=raw_input.get("air_quality"),
                )
                processed_reading, is_anomaly = self.analytics_provider.add_reading(
                    reading
                )

                raw_input["temperature"] = processed_reading.temperature
                raw_input["humidity"] = processed_reading.humidity
                raw_input["air_quality"] = processed_reading.air_quality

                if is_anomaly:
                    raw_input["anomaly_detected"] = True
            except Exception as e:
                logging.error(f"Error processing data through analytics: {e}")

        parts = []
        if raw_input.get("temperature") is not None:
            parts.append(f"Temperature: {raw_input['temperature']}°C")
        if raw_input.get("humidity") is not None:
            parts.append(f"Humidity: {raw_input['humidity']}%")
        if raw_input.get("air_quality") is not None:
            parts.append(f"Air Quality Index: {raw_input['air_quality']}")

        if not parts:
            return None

        message_text = "Environmental conditions: " + ", ".join(parts) + "."

        if raw_input.get("anomaly_detected"):
            message_text += " [ANOMALY DETECTED]"

        return Message(timestamp=time.time(), message=message_text)

    async def raw_to_text(self, raw_input: Optional[dict]):
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1].message}
// END
"""
        self.io_provider.add_input(
            self.descriptor_for_LLM, self.messages[-1].message, time.time()
        )
        self.messages = []
        return result
