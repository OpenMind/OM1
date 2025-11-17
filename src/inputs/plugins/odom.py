import asyncio
import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List, Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.odom_provider import OdomProvider, RobotState


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


class Odom(FuserInput[str]):
    """
    Odom input handler.

    A class that processes odometry inputs and generates text descriptions.
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

        logging.info(f"Config: {self.config}")

        use_zenoh = getattr(self.config, "use_zenoh", False)
        self.URID = getattr(config, "URID", "")
        unitree_ethernet = getattr(config, "unitree_ethernet", None)
        if use_zenoh:
            # probably a turtlebot
            logging.info(f"Odom using Zenoh and URID: {self.URID}")

        self.odom = OdomProvider(self.URID, use_zenoh, unitree_ethernet)
        self.descriptor_for_LLM = "Information about your location and body pose, to help plan your movements."

    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the Odom Provider.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[dict]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.1)

        try:
            return self.odom.position
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: dict) -> Message:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : list
            Raw input to be processed

        Returns
        -------
        Message
            A timestamped message containing the processed input
        """
        logging.debug(f"odom: {raw_input}")

        res = ""
        moving = raw_input["moving"]
        attitude = raw_input["body_attitude"]

        if attitude is RobotState.SITTING:
            res = "You are sitting down - do not generate new movement commands. "
        elif moving:
            # already moving
            res = "You are moving - do not generate new movement commands. "
        else:
            res = "You are standing still - you can move if you want to. "

        return Message(timestamp=time.time(), message=res)

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[dict]
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
