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
    topic: str = Field(default="rgb_image", description="Zenoh topic for the image stream")
    decode_format: str = Field(default="RAW", description="Image decode format hint")
    model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model id; supported: "
        "gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, "
        "gemini-3-flash-preview, gemini-3-pro-preview, "
        "gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview",
    )
    max_tokens: int = Field(default=1024, description="Token budget per VLM call")
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt sent with each frame. Defaults to a one-sentence "
        "scene-description prompt; override for task-specific use.",
    )


class VLMGeminiZenoh(FuserInput[VLMGeminiZenohConfig, Optional[str]]):
    """Gemini VLM input that subscribes to a Zenoh image topic. The
    counterpart to VLMVilaZenoh, but routes through the Gemini HTTP
    endpoint instead of the Vila WS.
    """

    def __init__(self, config: VLMGeminiZenohConfig):
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
        if content:
            logging.info(f"VLM Gemini (Zenoh) received message: {content}")
            self.message_buffer.put(content)
        else:
            logging.warning("VLM Gemini (Zenoh) received empty message")

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.5)
        try:
            return self.message_buffer.get_nowait()
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Append a formatted VLM message to the input buffer."""
        if raw_input is None:
            return
        msg = await self._raw_to_text(raw_input)
        if msg is not None:
            self.messages.append(msg)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return and clear the most recent formatted VLM message."""
        if not self.messages:
            return None
        latest = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n" "// START\n" f"{latest.message}\n" "// END\n"
        self.io_provider.add_input(self.__class__.__name__, latest.message, latest.timestamp)
        self.messages = []
        return result

    def stop(self):
        """Stop the VLM provider and release its resources."""
        if self.vlm:
            self.vlm.stop()
