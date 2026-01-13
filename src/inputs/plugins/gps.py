import asyncio
import logging
import time
from queue import Empty
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.gps_provider import GpsProvider
from providers.io_provider import IOProvider


class Gps(FuserInput[SensorConfig, Optional[dict]]):
    """
    GPS input handler for reading GPS and magnetometer data.

    This class processes GPS location data from the GpsProvider and converts
    it into formatted text messages for the LLM. It handles GPS coordinates
    (latitude, longitude, altitude) and quality indicators, converting them
    into human-readable location descriptions.

    The GPS input is essential for robot localization, navigation, and
    location-aware behavior planning. It provides rough GPS coordinates
    that can be used for outdoor navigation and spatial awareness.

    Attributes
    ----------
    gps : GpsProvider
        The GPS provider instance for accessing GPS data
    io_provider : IOProvider
        Provider for input/output operations
    messages : list[Message]
        Buffer for storing processed GPS messages
    descriptor_for_LLM : str
        Description string for the LLM ("GPS Location")
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize the GPS input handler.

        Parameters
        ----------
        config : SensorConfig
            Configuration object for the sensor input. Contains sensor-specific
            settings and parameters required for GPS data processing.

        Notes
        -----
        The initialization automatically creates a GpsProvider instance and
        an IOProvider instance. The GPS provider connects to the GPS hardware
        or data source to retrieve location data.
        """
        super().__init__(config)

        self.gps = GpsProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "GPS Location"

    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the GPS Provider.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[dict]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)

        try:
            return self.gps.data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw input to be processed

        Returns
        -------
        Message
            A timestamped message containing the processed input
        """
        logging.debug(f"gps: {raw_input}")

        d = raw_input
        if d:
            logging.debug(f"GPS Provider: {d}")
            lat = d["gps_lat"]
            lon = d["gps_lon"]
            alt = d["gps_alt"]
            qua = d["gps_qua"]

            lat_string = "South"
            if lat > 0:
                lat_string = "North"
            else:
                lat *= -1.0

            lon_string = "West"
            if lon > 0:
                lon_string = "East"
            else:
                lon *= -1.0

            if qua > 0:
                msg = f"Your rough GPS location is {lat} {lat_string}, {lon} {lon_string} at {alt}m altitude. "
                return Message(timestamp=time.time(), message=msg)
            else:
                return None
        else:
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw input to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with timestamp and class name,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
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
