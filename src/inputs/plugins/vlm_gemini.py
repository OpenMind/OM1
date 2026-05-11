import asyncio
import logging
import time
from queue import Empty, Queue
from typing import List, Optional, Union

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.vlm_gemini_provider import VLMGeminiProvider
from providers.vlm_gemini_zenoh_provider import VLMGeminiZenohProvider


class VLMGeminiConfig(SensorConfig):
    """
    Configuration for VLM Gemini Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        OM portal key. If unset, falls back to env $OM_API_KEY.
    base_url : str
        OM Gemini proxy URL (HTTP, not WS).
    topic : str
        Zenoh topic carrying sensor_msgs/Image frames.
    decode_format : str
        Stored on the VideoZenohStream but unused by it; safe to leave default.
    model : str
        Gemini model id.
    max_tokens : int
        Token budget. Reasoning models burn through this — bump if cut off.
    prompt : Optional[str]
        Prompt sent with each frame. Defaults to a one-sentence scene-description prompt; override for task-specific use.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="https://api.openmind.com/api/core/gemini",
        description="Base URL for the Gemini service",
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    camera_index: int = Field(default=0, description="Index of the camera device")

    topic: str = Field(default="camera/go2/image_raw", description="Zenoh topic for the image stream")
    decode_format: str = Field(default="RAW", description="Image decode format hint")
    use_sim: bool = Field(default=False, description="Whether to use the simulation stream endpoint")

    model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model id; supported (server-side): "
        "gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, "
        "gemini-3-flash-preview, gemini-3-pro-preview, "
        "gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview",
    )
    max_tokens: int = Field(
        default=1024,
        description="Token budget for VLM response. Reasoning-capable models "
        "(gemini-3.x, 2.5-pro) consume hidden reasoning tokens "
        "before visible content — bump to 2048+ if responses cut off.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt sent with each frame. Defaults to a one-sentence "
        "scene-description prompt; override for task-specific use.",
    )


class VLMGemini(FuserInput[VLMGeminiConfig, Optional[str]]):
    """
    Vision Language Model input handler.

    A class that processes image inputs and generates text descriptions using
    a vision language model. It maintains an internal buffer of processed messages
    and interfaces with a VLM provider for image analysis.

    The class handles asynchronous processing of images, maintains message history,
    and provides formatted output of the latest processed messages.
    """

    def __init__(self, config: VLMGeminiConfig):
        """
        Initialize VLM input handler.

        Sets up the required providers and buffers for handling VLM processing.
        Initializes connection to the VLM service and registers message handlers.

        Parameters
        ----------
        config : VLMGeminiConfig
            Configuration for the VLM input handler.
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
            self.config.stream_base_url or f"wss://api.openmind.com/api/core/teleops/stream/video?api_key={api_key}"
        )
        camera_index = self.config.camera_index

        provider_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.prompt is not None:
            provider_kwargs["prompt"] = self.config.prompt

        if self.config.use_sim:
            self.vlm: Union[VLMGeminiZenohProvider, VLMGeminiProvider] = VLMGeminiZenohProvider(
                base_url=self.config.base_url,
                api_key=api_key,
                topic=self.config.topic,
                decode_format=self.config.decode_format,
                **provider_kwargs,
            )
        else:
            self.vlm = VLMGeminiProvider(
                base_url=base_url,
                api_key=api_key,
                stream_url=stream_base_url,
                camera_index=camera_index,
                **provider_kwargs,
            )

        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

        self.descriptor_for_LLM = "Vision"

    def _handle_vlm_message(self, content: str):
        """
        Process incoming VLM messages.

        Parameters
        ----------
        content : str
            Plain text content from the VLM proxy (already extracted from
            choices[0].message.content by the provider).
        """
        if content:
            logging.info(f"VLM Gemini received message: {content}")
            self.message_buffer.put(content)
        else:
            logging.warning("VLM Gemini received empty message")

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
{self.descriptor_for_LLM}: "{latest_message.message}"
"""

        self.io_provider.add_input(self.__class__.__name__, latest_message.message, latest_message.timestamp)
        self.messages = []

        return result

    def stop(self):
        """
        Stop the VLM input.
        """
        if self.vlm:
            self.vlm.deregister_message_callback()
            self.vlm.stop()
