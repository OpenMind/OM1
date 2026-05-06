import asyncio
import logging
import os
import time
from queue import Empty, Queue
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.vlm_gemini_zenoh_provider import VLMGeminiZenohProvider


class VLMGeminiZenohConfig(SensorConfig):
    """
    Configuration for the Zenoh-sourced Gemini VLM input.

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
    """

    api_key: Optional[str] = Field(default=None, description="API Key (defaults to $OM_API_KEY)")
    base_url: str = Field(
        default="https://api.openmind.com/api/core/gemini",
        description="Base URL for the Gemini proxy",
    )
    topic: str = Field(default="camera/go2/image_raw", description="Zenoh topic for the image stream")
    decode_format: str = Field(default="RAW", description="Image decode format hint")
    model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model id; supported: "
        "gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, "
        "gemini-3-flash-preview, gemini-3-pro-preview, "
        "gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview",
    )
    max_tokens: int = Field(default=2048, description="Token budget per VLM call")
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt sent with each frame. Defaults to a one-sentence "
        "scene-description prompt; override for task-specific use.",
    )


class VLMGeminiZenoh(FuserInput[VLMGeminiZenohConfig, Optional[str]]):
    """
    Vision Language Model input handler.

    A class that processes image inputs and generates text descriptions using
    a vision language model. It maintains an internal buffer of processed messages
    and interfaces with a VLM provider for image analysis.

    The class handles asynchronous processing of images, maintains message history,
    and provides formatted output of the latest processed messages.
    """

    def __init__(self, config: VLMGeminiZenohConfig):
        """
        Initialize the provider, set up the VLM provider, and prepare for message handling.

        Parameters
        ----------
        config : VLMGeminiZenohConfig
            Configuration for the provider.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()

        api_key = self.config.api_key or os.environ.get("OM_API_KEY", "")
        if not api_key:
            raise ValueError("VLMGeminiZenoh: api_key not configured and OM_API_KEY env var is empty")

        if self.config.prompt is not None:
            self.vlm: VLMGeminiZenohProvider = VLMGeminiZenohProvider(
                base_url=self.config.base_url,
                api_key=api_key,
                topic=self.config.topic,
                decode_format=self.config.decode_format,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                prompt=self.config.prompt,
            )
        else:
            self.vlm: VLMGeminiZenohProvider = VLMGeminiZenohProvider(
                base_url=self.config.base_url,
                api_key=api_key,
                topic=self.config.topic,
                decode_format=self.config.decode_format,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
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
            logging.info(f"VLM Gemini (Zenoh) received message: {content}")
            self.message_buffer.put(content)
        else:
            logging.warning("VLM Gemini (Zenoh) received empty message")

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
            return self.message_buffer.get_nowait()
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
        if not self.messages:
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
            self.vlm.stop()
