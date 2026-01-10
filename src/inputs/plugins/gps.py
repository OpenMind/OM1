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
    Input plugin for processing GPS and magnetometer sensor data.

    This input plugin continuously polls a GpsProvider instance for location
    and orientation data from a connected GPS device. The plugin converts raw
    GPS coordinates and magnetometer readings into natural language messages
    that can be processed by the agent's language model for location-aware
    decision making.

    The GPS input operates asynchronously, polling the GPS provider every 0.5
    seconds for new position data. When valid GPS data is available (quality
    indicator > 0), the plugin generates formatted location messages including
    latitude, longitude, altitude, and cardinal directions. These messages are
    buffered and made available to the agent's input pipeline for contextual
    awareness during autonomous operations.

    The plugin maintains an internal message buffer and integrates with the
    IOProvider to track all GPS inputs for logging and analysis purposes.

    Typical use cases include:
    - Providing location context to the agent for navigation decisions
    - Enabling location-aware task planning and execution
    - Supporting geofencing and boundary awareness
    - Facilitating waypoint navigation and path following
    - Integrating GPS data into the agent's decision-making process

    The GPS input converts technical coordinate data into human-readable
    descriptions (e.g., "Your rough GPS location is 37.7749 North, 122.4194
    West at 10m altitude") that the language model can understand and reason
    about when making navigation or location-based decisions.

    Notes
    -----
    The GPS device must provide valid position fixes (quality > 0) for location
    messages to be generated. Poor GPS signal or indoor operation may result in
    no location updates being available to the agent.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize GPS input plugin with configuration.

        Sets up the GPS provider connection, initializes the message buffer,
        and configures the input descriptor for the language model.

        Parameters
        ----------
        config : SensorConfig
            Configuration object for the sensor input. Uses the base SensorConfig
            as GPS input does not require additional configuration parameters beyond
            the standard sensor settings.

        Notes
        -----
        The GPS provider is initialized with default settings and will attempt to
        connect to the configured GPS hardware. The descriptor "GPS Location" is
        used to label this input source in the agent's context.
        """
        super().__init__(config)

        self.gps = GpsProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "GPS Location"

    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the GPS Provider.

        Checks the GPS provider for updated position data with a brief delay
        to prevent excessive CPU usage. This method is called continuously by
        the input polling loop.

        Returns
        -------
        Optional[dict]
            Dictionary containing GPS data with keys 'gps_lat', 'gps_lon',
            'gps_alt', and 'gps_qua' (quality indicator), or None if no data
            is available.

        Notes
        -----
        The 0.5 second sleep interval balances responsiveness with system
        resource usage, as GPS positions typically update at 1Hz or slower.
        """
        await asyncio.sleep(0.5)

        try:
            return self.gps.data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Process raw GPS data to generate a natural language message.

        Converts raw GPS coordinates (latitude, longitude, altitude) into a
        human-readable location description with cardinal directions. Only
        generates messages when GPS quality indicator is positive.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw GPS data dictionary containing 'gps_lat', 'gps_lon', 'gps_alt',
            and 'gps_qua' keys, or None if no data is available.

        Returns
        -------
        Optional[Message]
            A timestamped message containing the formatted location description,
            or None if the input is invalid or GPS quality is insufficient.

        Notes
        -----
        The method converts latitude/longitude to absolute values with cardinal
        directions (North/South, East/West) to create more natural language
        descriptions. Negative latitudes indicate South, negative longitudes
        indicate West.
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
        Update message buffer with processed GPS data.

        Processes raw GPS input and appends the resulting message to the
        internal buffer if valid data is available.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw GPS data to be processed.

        Notes
        -----
        This method is called by the input processing pipeline and handles
        buffering of messages for later retrieval by the agent.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent GPS location message from the buffer,
        formats it with the input descriptor for the language model, logs
        it to the IO provider, and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest location message with
            INPUT markers and descriptor, or None if the buffer is empty.

        Notes
        -----
        The formatted output includes START/END markers to clearly delineate
        the GPS input in the agent's context. After formatting, the message
        is logged to the IO provider for tracking and the buffer is cleared.
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
