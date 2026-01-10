import asyncio
import logging
import time
from queue import Empty
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.rtk_provider import RtkProvider


class Rtk(FuserInput[SensorConfig, Optional[dict]]):
    """
    Input plugin for processing Real-Time Kinematic (RTK) GPS sensor data.

    This input plugin continuously polls an RtkProvider instance for high-precision
    positioning data from a connected RTK GPS receiver. The plugin converts raw
    RTK coordinates into natural language messages that provide centimeter-level
    location accuracy for the agent's language model, enabling precise navigation
    and positioning decisions.

    RTK GPS achieves significantly higher accuracy than standard GPS by using
    correction data from a base station. This input plugin operates asynchronously,
    polling the RTK provider every 0.5 seconds for position updates. When valid
    RTK data is available (quality indicator > 0), the plugin generates formatted
    precision location messages including latitude, longitude, altitude, and
    cardinal directions.

    The plugin maintains an internal message buffer and integrates with the
    IOProvider to track all RTK inputs for logging and analysis. The high
    precision of RTK positioning (typically 1-2 cm accuracy) makes this input
    essential for applications requiring accurate localization and path following.

    Typical use cases include:
    - Providing precise location context for autonomous navigation
    - Enabling accurate waypoint tracking and path following
    - Supporting precision agriculture and field operations
    - Facilitating accurate docking and landing maneuvers
    - Enabling centimeter-level geofencing and boundary detection
    - Integrating high-precision positioning into decision-making

    The RTK input converts technical coordinate data into human-readable
    descriptions (e.g., "Your precise location is 37.7749 North, 122.4194
    West at 10m altitude") that emphasize the high accuracy of the positioning
    data, allowing the language model to understand and leverage the precision
    when planning navigation tasks.

    Notes
    -----
    The RTK receiver must be receiving correction data from a base station to
    achieve centimeter-level accuracy. Without corrections, the system will
    fall back to standard GPS accuracy. Valid position fixes (quality > 0) are
    required for location messages to be generated.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize RTK input plugin with configuration.

        Sets up the RTK provider connection, initializes the message buffer,
        and configures the input descriptor for the language model.

        Parameters
        ----------
        config : SensorConfig
            Configuration object for the sensor input. Uses the base SensorConfig
            as RTK input does not require additional configuration parameters beyond
            the standard sensor settings.

        Notes
        -----
        The RTK provider is initialized with default settings and will attempt to
        connect to the configured RTK GPS hardware. The descriptor "Precision
        Location" is used to label this input source in the agent's context,
        emphasizing the high-accuracy nature of RTK positioning data.
        """
        super().__init__(config)

        self.rtk = RtkProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Precision Location"

    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the RTK Provider.

        Checks the RTK provider for updated high-precision position data with
        a brief delay to prevent excessive CPU usage. This method is called
        continuously by the input polling loop.

        Returns
        -------
        Optional[dict]
            Dictionary containing RTK data with keys 'rtk_lat', 'rtk_lon',
            'rtk_alt', and 'rtk_qua' (quality indicator), or None if no data
            is available.

        Notes
        -----
        The 0.5 second sleep interval balances responsiveness with system
        resource usage. RTK systems typically provide position updates at
        1-10Hz, so this polling rate is sufficient for most applications.
        """
        await asyncio.sleep(0.5)

        try:
            return self.rtk.data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Process raw RTK data to generate a natural language message.

        Converts raw RTK coordinates (latitude, longitude, altitude) into a
        human-readable precision location description with cardinal directions.
        Only generates messages when RTK quality indicator is positive,
        indicating valid correction data is being received.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw RTK data dictionary containing 'rtk_lat', 'rtk_lon', 'rtk_alt',
            and 'rtk_qua' keys, or None if no data is available.

        Returns
        -------
        Optional[Message]
            A timestamped message containing the formatted precision location
            description, or None if the input is invalid or RTK quality is
            insufficient.

        Notes
        -----
        The method converts latitude/longitude to absolute values with cardinal
        directions (North/South, East/West) to create natural language descriptions.
        The term "precise location" is used to emphasize the high accuracy of RTK
        positioning compared to standard GPS. Negative latitudes indicate South,
        negative longitudes indicate West.
        """
        logging.debug(f"rtk: {raw_input}")

        r = raw_input
        if r:
            logging.debug(f"RTK Provider: {r}")
            lat = r["rtk_lat"]
            lon = r["rtk_lon"]
            alt = r["rtk_alt"]
            qua = r["rtk_qua"]

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
                msg = f"Your precise location is {lat} {lat_string}, {lon} {lon_string} at {alt}m altitude. "
                return Message(timestamp=time.time(), message=msg)
            else:
                return None
        else:
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer with processed RTK data.

        Processes raw RTK input and appends the resulting message to the
        internal buffer if valid high-precision data is available.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw RTK data to be processed.

        Notes
        -----
        This method is called by the input processing pipeline and handles
        buffering of precision location messages for later retrieval by the agent.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent RTK precision location message from the buffer,
        formats it with the input descriptor for the language model, logs it to
        the IO provider, and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest precision location message
            with INPUT markers and descriptor, or None if the buffer is empty.

        Notes
        -----
        The formatted output includes START/END markers to clearly delineate
        the RTK input in the agent's context. After formatting, the message is
        logged to the IO provider for tracking and the buffer is cleared. The
        "Precision Location" descriptor helps the language model distinguish
        high-accuracy RTK data from standard GPS data.
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
