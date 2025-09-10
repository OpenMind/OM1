import asyncio
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.rplidar_provider import RPLidarProvider


@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    message : str
        Content of the message
    """

    timestamp: float
    message: str


class RPLidar(FuserInput[str]):
    """
    RPLidar input handler.

    A class that processes RPLidar inputs and generates text descriptions.
    It maintains an internal buffer of processed messages.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Build lidar configuration from config
        lidar_config = self._extract_lidar_config(config)

        # Initialize RPLidar Provider
        self.lidar: RPLidarProvider = RPLidarProvider(**lidar_config)
        self.lidar.start()

        self.descriptor_for_LLM = "Information about objects and walls around you, to plan your movements and avoid bumping into things."

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages from the RPLidar Provider.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[str]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.2)

        try:
            return self.lidar.lidar_string
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: str) -> Message:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : str
            Raw input string to be processed

        Returns
        -------
        Message
            A timestamped message containing the processed input
        """
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent message from the buffer, formats it
        with timestamp and class name, adds it to the IO provider,
        and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest message and metadata,
            or None if the buffer is empty

        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    def _extract_lidar_config(self, config: SensorConfig) -> dict:
        """Extract lidar configuration parameters from sensor config."""
        lidar_config = {
            "serial_port": getattr(config, "serial_port", None),
            "use_zenoh": getattr(config, "use_zenoh", False),
            "half_width_robot": getattr(config, "half_width_robot", 0.20),
            "angles_blanked": getattr(config, "angles_blanked", []),
            "relevant_distance_max": getattr(config, "relevant_distance_max", 1.1),
            "relevant_distance_min": getattr(config, "relevant_distance_min", 0.08),
            "sensor_mounting_angle": getattr(config, "sensor_mounting_angle", 180.0),
            "URID": getattr(config, "URID", ""),
            "multicast_address": getattr(config, "multicast_address", ""),
            "machine_type": getattr(config, "machine_type", "go2"),
            "log_file": getattr(config, "log_file", False),
        }

        return lidar_config
