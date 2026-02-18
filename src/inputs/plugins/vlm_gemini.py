import logging
from typing import Optional

from openai.types.chat import ChatCompletion
from pydantic import Field

from inputs.base import SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput
from providers.vlm_gemini_provider import VLMGeminiProvider


class VLMGeminiConfig(SensorConfig):
    """Configuration for VLM Gemini Sensor."""

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="https://api.openmind.org/api/core/gemini",
        description="Base URL for the Gemini service",
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    camera_index: int = Field(default=0, description="Index of the camera device")


class VLMGemini(BaseVLMFuserInput[VLMGeminiConfig]):
    """VLM input handler using Gemini API with local camera."""

    def __init__(self, config: VLMGeminiConfig):
        super().__init__(config)

        api_key = self.config.api_key
        if api_key is None or api_key == "":
            raise ValueError("config file missing api_key")

        stream_base_url = (
            self.config.stream_base_url
            or f"wss://api.openmind.org/api/core/teleops/stream/video?api_key={api_key}"
        )

        self.vlm: VLMGeminiProvider = VLMGeminiProvider(
            base_url=self.config.base_url,
            api_key=api_key,
            stream_url=stream_base_url,
            camera_index=self.config.camera_index,
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

    def _handle_vlm_message(self, raw_message: ChatCompletion) -> None:
        """Extract content from Gemini ChatCompletion response."""
        content = raw_message.choices[0].message.content
        if content is not None:
            logging.info(f"VLM Gemini received message: {content}")
            self.message_buffer.put(content)
        else:
            logging.warning("VLM Gemini received message with None content")
