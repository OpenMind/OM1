import asyncio
import json
import logging
import time
from queue import Empty, Queue
from typing import Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.turtlebot4_camera_vlm_provider import TurtleBot4CameraVLMProvider


class TurtleBot4CameraVLMCloudConfig(SensorConfig):
    """
    Configuration for TurtleBot4 Camera VLM Cloud Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    base_url : str
        Base URL for the VLM service.
    stream_base_url : Optional[str]
        Stream Base URL.
    URID : str
        URID (Unitree ID).
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="wss://api-vila.openmind.org",
        description="Base URL for the VLM service",
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    URID: str = Field(default="default", description="URID (Unitree ID)")


class TurtleBot4CameraVLMCloud(
    FuserInput[TurtleBot4CameraVLMCloudConfig, Optional[str]]
):
    """
    TurtleBot4 Camera VLM bridge.

    Takes TurtleBot4 Camera images, sends them to a cloud VLM provider,
    converts the responses to text strings, and sends them to the fuser.
    """

    def __init__(self, config: TurtleBot4CameraVLMCloudConfig):
        """
        Initialize VLM input handler.

        Sets up the required providers and buffers for handling VLM processing.
        Initializes connection to the VLM service and registers message handlers.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        self.descriptor_for_LLM = "Vision"

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize VLM provider
        api_key = self.config.api_key

        base_url = self.config.base_url
        stream_base_url = (
            self.config.stream_base_url
            or f"wss://api.openmind.org/api/core/teleops/stream/video?api_key={api_key}"
        )
        URID = self.config.URID

        self.vlm: TurtleBot4CameraVLMProvider = TurtleBot4CameraVLMProvider(
            ws_url=base_url, URID=URID, stream_url=stream_base_url
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

    def _handle_vlm_message(self, raw_message: Optional[str]):
        """
        Process incoming VLM messages.

        Parses JSON messages from the VLM service and adds valid responses
        to the message buffer for further processing.

        Parameters
        ----------
        raw_message : Optional[str]
            Raw JSON message received from the VLM service
        """
        if raw_message is None:
            return

        try:
            json_message: Dict = json.loads(raw_message)
            if "vlm_reply" in json_message:
                vlm_reply = json_message["vlm_reply"]
                self.message_buffer.put(vlm_reply)
                logging.info("Detected VLM message: %s", vlm_reply)
        except json.JSONDecodeError:
            pass

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages from the VLM service.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[str]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)
        try:
            message = self.message_buffer.get_nowait()
            return message
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

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
