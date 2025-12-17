import asyncio
import logging
import time
from queue import Empty, Queue
from typing import List, Optional

from openai.types.chat import ChatCompletion
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.vlm_openai_provider import VLMOpenAIProvider


class VLMOpenAIConfig(SensorConfig):
    """
    Configuration for VLM OpenAI Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    base_url : str
        Base URL for the OpenAI service.
    stream_base_url : Optional[str]
        Stream Base URL.
    camera_index : int
        Index of the camera device.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="https://api.openmind.org/api/core/openai", description="Base URL"
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    camera_index: int = Field(default=0, description="Camera Index")


class VLMOpenAI(FuserInput[VLMOpenAIConfig, Optional[str]]):
    """
    Vision Language Model input handler.

    A class that processes image inputs and generates text descriptions using
    a vision language model. It maintains an internal buffer of processed messages
    and interfaces with a VLM provider for image analysis.

    The class handles asynchronous processing of images, maintains message history,
    and provides formatted output of the latest processed messages.
    """

    def __init__(self, config: VLMOpenAIConfig):
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
        api_key = self.config.api_key

        if api_key is None or api_key == "":
            raise ValueError("config file missing api_key")

        base_url = self.config.base_url
        stream_base_url = (
            self.config.stream_base_url
            or f"wss://api.openmind.org/api/core/teleops/stream/video?api_key={api_key}"
        )
        camera_index = self.config.camera_index

        self.vlm: VLMOpenAIProvider = VLMOpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            stream_url=stream_base_url,
            camera_index=camera_index,
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

        self.descriptor_for_LLM = "Vision"

    def _handle_vlm_message(self, raw_message: ChatCompletion):
        """
        Process incoming VLM messages.

        Parses JSON messages from the VLM service and adds valid responses
        to the message buffer for further processing.

        Parameters
        ----------
        raw_message : str
            Raw JSON message received from the VLM service
        """
        logging.info(f"VLM OpenAI received message: {raw_message}")
        content = raw_message.choices[0].message.content
        if content is not None:
            self.message_buffer.put(content)

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
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
