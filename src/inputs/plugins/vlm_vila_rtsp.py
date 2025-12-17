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
from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider


class VLMVilaRTSPConfig(SensorConfig):
    """
    Configuration for VLM Vila RTSP Sensor.

    Parameters
    ----------
    base_url : str
        Base URL for the VLM service.
    rtsp_url : str
        RTSP URL for the camera stream.
    decode_format : str
        Image decode format (e.g., "H264").
    """

    base_url: str = Field(
        default="wss://api-vila.openmind.org",
        description="Base URL for the VLM service",
    )
    rtsp_url: str = Field(
        default="rtsp://localhost:8554/top_camera",
        description="RTSP URL for the camera stream",
    )
    decode_format: str = Field(
        default="H264", description='Image decode format (e.g., "H264")'
    )


class VLMVilaRTSP(FuserInput[VLMVilaRTSPConfig, Optional[str]]):
    """
    Vision Language Model input handler.

    A class that processes image inputs and generates text descriptions using
    a vision language model. It maintains an internal buffer of processed messages
    and interfaces with a VLM provider for image analysis.

    The class handles asynchronous processing of images, maintains message history,
    and provides formatted output of the latest processed messages.
    """

    def __init__(self, config: VLMVilaRTSPConfig):
        """
        Initialize VLM input handler.

        Sets up the required providers and buffers for handling VLM processing.
        Initializes connection to the VLM service and registers message handlers.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize VLM provider
        base_url = self.config.base_url
        rtsp_url = self.config.rtsp_url
        decode_format = self.config.decode_format

        self.vlm: VLMVilaRTSPProvider = VLMVilaRTSPProvider(
            ws_url=base_url, rtsp_url=rtsp_url, decode_format=decode_format
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

        self.descriptor_for_LLM = "Vision"

    def _handle_vlm_message(self, raw_message: str):
        """
        Process incoming VLM messages.

        Parses JSON messages from the VLM service and adds valid responses
        to the message buffer for further processing.

        Parameters
        ----------
        raw_message : str
            Raw JSON message received from the VLM service
        """
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
