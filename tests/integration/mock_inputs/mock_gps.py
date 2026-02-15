import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from inputs.plugins.gps import Gps
from providers.io_provider import IOProvider
from tests.integration.mock_inputs.data_providers.mock_state_provider import (
    get_next_gps,
)


class MockGps(Gps):
    """
    Mock implementation of Gps that uses the central state provider.

    This class bypasses the GpsProvider and serial port hardware to inject
    mock GPS data for integration testing.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize with mock state provider, bypassing real hardware.

        Parameters
        ----------
        config : SensorConfig, optional
            Configuration for the sensor
        """
        # Skip Gps.__init__ to avoid GpsProvider serial port setup
        FuserInput.__init__(self, config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "MOCK GPS Location (Integration Test)"

        self.running = True
        self.data_processed = False

        logging.info("MockGps initialized - using mock state provider")

    async def _poll(self) -> Optional[dict]:
        """
        Poll for mock GPS data from the state provider.

        Returns
        -------
        Optional[dict]
            GPS data dict with gps_lat, gps_lon, gps_alt, gps_qua keys, or None
        """
        data = get_next_gps()
        if data is not None:
            logging.info(
                f"MockGps: GPS lat={data.get('gps_lat')}, lon={data.get('gps_lon')}"
            )
            return data

        if not self.data_processed:
            logging.info("MockGps: No more GPS data to process")
            self.data_processed = True

        return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """Process raw GPS data to generate a timestamped message."""
        if raw_input is None:
            return None

        lat = raw_input.get("gps_lat", 0.0)
        lon = raw_input.get("gps_lon", 0.0)
        alt = raw_input.get("gps_alt", 0.0)
        qua = raw_input.get("gps_qua", 0)

        lat_string = "North" if lat > 0 else "South"
        lon_string = "East" if lon > 0 else "West"

        display_lat = abs(lat)
        display_lon = abs(lon)

        if qua > 0:
            msg = f"Your rough GPS location is {display_lat} {lat_string}, {display_lon} {lon_string} at {alt}m altitude. "
            return Message(timestamp=time.time(), message=msg)

        return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """Update message buffer with GPS text."""
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
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

    def stop(self):
        """Stop the mock GPS input."""
        self.running = False
        logging.info("MockGps: Stopped")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logging.info("MockGps: Cleanup completed")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
