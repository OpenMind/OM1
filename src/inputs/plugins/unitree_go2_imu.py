import asyncio
import logging
import math
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import IOProvider

try:
    from unitree.unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_  # type: ignore
except ImportError:
    logging.warning(
        "Unitree SDK not found. Please install the Unitree SDK to use this plugin."
    )

    class ChannelSubscriber:
        """
        Placeholder for ChannelSubscriber when Unitree SDK is not installed.
        """

        def __init__(self):
            pass

    class LowState_:
        """
        Placeholder for LowState_ when Unitree SDK is not installed.
        """

        def __init__(self):
            pass


class UnitreeGo2IMUConfig(SensorConfig):
    """
    Configuration for Unitree Go2 IMU fall detection sensor.

    Parameters
    ----------
    fall_threshold_deg : float
        Tilt angle in degrees that indicates a fall.
    warning_threshold_deg : float
        Tilt angle in degrees that indicates dangerous tilting.
    api_key : Optional[str]
        API Key.
    """

    fall_threshold_deg: float = Field(
        default=45.0,
        description="Tilt angle in degrees that indicates a fall.",
    )
    warning_threshold_deg: float = Field(
        default=30.0,
        description="Tilt angle in degrees that indicates dangerous tilting.",
    )
    api_key: Optional[str] = Field(default=None, description="API Key")


class UnitreeGo2IMU(FuserInput[UnitreeGo2IMUConfig, Optional[str]]):
    """
    Unitree Go2 IMU fall detection sensor.

    Subscribes to Unitree CycloneDDS LowState messages and monitors
    roll/pitch angles from IMU data to detect falls and dangerous tilting.

    Maintains a buffer of processed messages.
    """

    def __init__(self, config: UnitreeGo2IMUConfig):
        """
        Initialize IMU fall detection sensor.

        Parameters
        ----------
        config : UnitreeGo2IMUConfig
            Configuration settings for the sensor input.
        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.messages: list[Message] = []

        self.roll_deg = 0.0
        self.pitch_deg = 0.0

        self.lowstate_subscriber = None
        try:
            self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # type: ignore
            logging.info("IMU fall detection monitor initialized")
        except Exception as e:
            logging.error(f"Error initializing IMU fall detection monitor: {e}")

        if self.lowstate_subscriber:
            self.lowstate_subscriber.Init(self.low_state_handler, 10)  # type: ignore

        self.descriptor_for_LLM = "Body Orientation"

    def low_state_handler(self, msg: LowState_):
        """
        Handle incoming LowState messages from Unitree Go2.

        Extracts roll and pitch from IMU RPY data and converts to degrees.

        Parameters
        ----------
        msg : LowState_
            Incoming LowState message containing IMU data.
        """
        try:
            self.roll_deg = round(math.degrees(float(msg.imu_state.rpy[0])), 2)  # type: ignore
            self.pitch_deg = round(math.degrees(float(msg.imu_state.rpy[1])), 2)  # type: ignore
        except (AttributeError, IndexError, TypeError) as e:
            logging.warning(f"Incomplete IMU data in LowState message: {e}")
            self.roll_deg = 0.0
            self.pitch_deg = 0.0

    async def _poll(self) -> Optional[str]:
        """
        Poll for new IMU data and evaluate tilt status.

        Returns
        -------
        Optional[str]
            Alert message if tilt exceeds thresholds, None otherwise.
        """
        await asyncio.sleep(0.5)

        abs_roll = abs(self.roll_deg)
        abs_pitch = abs(self.pitch_deg)
        fall_threshold = self.config.fall_threshold_deg
        warning_threshold = self.config.warning_threshold_deg

        logging.debug(f"IMU roll: {self.roll_deg}, pitch: {self.pitch_deg}")

        if abs_roll > fall_threshold or abs_pitch > fall_threshold:
            return (
                f"CRITICAL: You have fallen over. "
                f"Roll: {self.roll_deg} degrees, Pitch: {self.pitch_deg} degrees. "
                f"Use the recover action to stand back up immediately."
            )

        if abs_roll > warning_threshold or abs_pitch > warning_threshold:
            return (
                f"WARNING: You are tilting dangerously. "
                f"Roll: {self.roll_deg} degrees, Pitch: {self.pitch_deg} degrees. "
                f"Try to stabilize yourself."
            )

        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw tilt status string to a timestamped Message.

        Parameters
        ----------
        raw_input : Optional[str]
            Alert message from _poll, or None if orientation is normal.

        Returns
        -------
        Optional[Message]
            Timestamped message if alert exists, None otherwise.
        """
        if raw_input is not None:
            return Message(timestamp=time.time(), message=raw_input)
        return None

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw IMU data to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Alert message from _poll.
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
            Formatted string of buffer contents or None if buffer is empty.
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
