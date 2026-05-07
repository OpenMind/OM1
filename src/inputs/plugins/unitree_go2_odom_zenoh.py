import asyncio
import time
from queue import Empty, Queue
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.odom_provider_base import RobotState
from providers.unitree_go2_odom_zenoh_provider import UnitreeGo2OdomZenohProvider


class UnitreeGo2OdomZenohConfig(SensorConfig):
    """
    Configuration for the Unitree Go2 Odom Zenoh Provider.

    Parameters
    ----------
    api_key : Optional[str]
        API key for authentication, if required by the Zenoh session.
    topic : str
        Zenoh keyexpression to subscribe to.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    topic: str = Field(
        default="utlidar/robot_pose",
        description="Zenoh key for Go2 robot_pose / odom.",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to use the simulation Zenoh endpoint instead of a local one.",
    )


class UnitreeGo2OdomZenoh(FuserInput[UnitreeGo2OdomZenohConfig, Optional[dict]]):
    """Unitree Go2 Zenoh Odom Provider."""

    def __init__(self, config: UnitreeGo2OdomZenohConfig):
        """
        Initialize the provider and start the background odometry subscriber process.

        Parameters
        ----------
        config : UnitreeGo2OdomZenohConfig
            Configuration for the provider.
        """
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()

        self.odom = UnitreeGo2OdomZenohProvider(
            api_key=self.config.api_key,
            topic=self.config.topic,
            use_sim=self.config.use_sim,
        )

        self.descriptor_for_LLM = "Information about your location and body pose, to help plan your movements."

    async def _poll(self) -> Optional[dict]:
        """
        Poll the latest odometry data from the provider.

        Returns
        -------
        Optional[dict]
            The latest odometry data as a dictionary, or None if no data is available.
        """
        await asyncio.sleep(0.1)
        try:
            return self.odom.position
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Convert raw odometry data into a human-readable message about the robot's state.

        Parameters
        ----------
        raw_input : Optional[dict]
            The raw odometry data as a dictionary.

        Returns
        -------
        Optional[Message]
            A Message object containing a human-readable description of the robot's state, or None if input
        """
        if raw_input is None:
            return None

        moving = raw_input["moving"]
        attitude = raw_input["body_attitude"]

        if attitude is RobotState.SITTING:
            res = "You are sitting down - do not generate new movement commands. "
        elif moving:
            res = "You are moving - do not generate new movement commands. "
        else:
            res = "You are standing still - you can move if you want to. "

        return Message(timestamp=time.time(), message=res)

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Convert raw odometry data into a human-readable message and store it in the message buffer.

        Parameters
        ----------
        raw_input : Optional[dict]
            The raw odometry data as a dictionary.
        """
        msg = await self._raw_to_text(raw_input)
        if msg is not None:
            if len(self.messages) == 0:
                self.messages.append(msg)
            else:
                self.messages[-1] = msg

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Get the most recent message as a formatted string for the LLM, and log it to the IOProvider.

        Returns
        -------
        Optional[str]
            A formatted string containing the latest message for the LLM, or None if no messages are
            available.
        """
        if not self.messages:
            return None

        latest = self.messages[-1]

        result = f"""
{self.descriptor_for_LLM}: "{latest.message}"
"""
        self.io_provider.add_input(self.__class__.__name__, latest.message, latest.timestamp)

        self.messages = []
        return result
