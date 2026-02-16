import asyncio
import time
from queue import Empty
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.gps_provider import GpsProvider
from providers.io_provider import IOProvider


class Gps(FuserInput[SensorConfig, Optional[dict]]):

    def __init__(self, config: SensorConfig):
        super().__init__(config)

        self.gps = GpsProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "GPS Location"

    async def _poll(self) -> Optional[dict]:
        await asyncio.sleep(0.5)

        try:
            return self.gps.data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        if not raw_input:
            return None

        try:
            lat = raw_input.get("gps_lat")
            lon = raw_input.get("gps_lon")
            alt = raw_input.get("gps_alt")
            qua = raw_input.get("gps_qua")

            # Validate numeric values
            if not isinstance(lat, (int, float)):
                return None
            if not isinstance(lon, (int, float)):
                return None
            if not isinstance(alt, (int, float)):
                return None
            if not isinstance(qua, (int, float)):
                return None

            # Poor GPS quality must return None (OM1 requirement)
            if qua <= 0:
                return None

            lat_dir = "North" if lat > 0 else "South"
            lon_dir = "East" if lon > 0 else "West"

            message_text = (
                f"Latitude: {abs(lat)} {lat_dir}, "
                f"Longitude: {abs(lon)} {lon_dir}, "
                f"Altitude: {alt}, "
                f"Quality: {qua}"
            )

            # IMPORTANT: OM1 Message constructor order
            return Message(time.time(), message_text)

        except Exception:
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            latest_message.message,
            latest_message.timestamp,
        )

        self.messages = []

        return result
