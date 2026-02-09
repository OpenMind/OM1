import asyncio
import time
from queue import Empty, Queue
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.unitree_go2_rplidar_provider import UnitreeGo2RPLidarProvider


class RPLidarConfig(SensorConfig):
    """
    Configuration for RPLidar Sensor.

    Parameters
    ----------
    serial_port : Optional[str]
        Serial Port to connect to.
    half_width_robot : float
        Half width of the robot.
    angles_blanked : List[float]
        List of angles to blank out.
    relevant_distance_max : float
        Maximum relevant distance.
    relevant_distance_min : float
        Minimum relevant distance.
    sensor_mounting_angle : float
        Mounting angle of the sensor.
    log_file : bool
        Whether to log to a file.
    """

    serial_port: Optional[str] = Field(
        default=None, description="Serial Port to connect to"
    )
    half_width_robot: float = Field(default=0.20, description="Half width of the robot")
    angles_blanked: List[float] = Field(
        default_factory=list, description="List of angles to blank out"
    )
    relevant_distance_max: float = Field(
        default=1.1, description="Maximum relevant distance"
    )
    relevant_distance_min: float = Field(
        default=0.08, description="Minimum relevant distance"
    )
    sensor_mounting_angle: float = Field(
        default=180.0, description="Mounting angle of the sensor"
    )
    log_file: bool = Field(default=False, description="Whether to log to a file")


class UnitreeGo2RPLidar(FuserInput[RPLidarConfig, Optional[str]]):
    """
    RPLidar input handler for processing laser range finder data.
    """

    def __init__(self, config: RPLidarConfig):
        """
        Initialize the RPLidar input handler.

        Parameters
        ----------
        config : RPLidarConfig
            Configuration for the RPLidar input handler.
        """
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
        self.lidar: UnitreeGo2RPLidarProvider = UnitreeGo2RPLidarProvider(
            **lidar_config
        )
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

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

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

    def _extract_lidar_config(self, config: RPLidarConfig) -> dict:
        """
        Extract lidar configuration parameters from sensor config.

        Converts the RPLidarConfig Pydantic model into a dictionary format
        suitable for passing to the RPLidarProvider constructor.

        Parameters
        ----------
        config : RPLidarConfig
            Configuration object containing all sensor parameters.

        Returns
        -------
        dict
            Dictionary containing all configuration parameters with their
            corresponding field names and values.
        """
        lidar_config = {
            "serial_port": config.serial_port,
            "half_width_robot": config.half_width_robot,
            "angles_blanked": config.angles_blanked,
            "relevant_distance_max": config.relevant_distance_max,
            "relevant_distance_min": config.relevant_distance_min,
            "sensor_mounting_angle": config.sensor_mounting_angle,
            "log_file": config.log_file,
        }

        return lidar_config
