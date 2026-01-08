import asyncio
import time
from typing import Dict, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class BME280Config(SensorConfig):
    """Configuration for the BME280 input plugin."""

    i2c_address: int = 0x76
    sampling_rate_hz: float = 1.0


class BME280Input(FuserInput[BME280Config, Optional[Dict]]):
    """Environmental sensor input plugin using the BME280 sensor."""

    def __init__(self, config: BME280Config):
        """Initialize the BME280 input plugin."""
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages = []
        self.descriptor_for_LLM = "Environmental sensor data from BME280"

    async def _poll(self) -> Optional[Dict]:
        """Poll the sensor and return environmental data."""
        await asyncio.sleep(1.0 / max(self.config.sampling_rate_hz, 0.1))

        # Mock data (CI-safe, no hardware dependency)
        return {
            "temperature_c": 25.0,
            "humidity_percent": 40.0,
            "pressure_hpa": 1013.25,
        }

    async def raw_to_text(self, raw_input):
        """Convert raw sensor data into a Message object."""
        if raw_input is None:
            return

        self.messages.append(Message(timestamp=time.time(), message=str(raw_input)))

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and return the latest buffered sensor message."""
        if not self.messages:
            return None

        msg = self.messages[-1]
        self.messages = []

        return (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n" f"{msg.message}\n// END\n"
        )
