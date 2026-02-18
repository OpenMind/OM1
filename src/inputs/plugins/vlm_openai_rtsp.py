import logging
from typing import Optional

from openai.types.chat import ChatCompletion
from pydantic import Field

from inputs.base import SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput
from providers.vlm_openai_rtsp_provider import VLMOpenAIRTSPProvider


class VLMOpenAIRTSPConfig(SensorConfig):
    """Configuration for VLM OpenAI RTSP Sensor."""

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="https://api.openmind.org/api/core/openai",
        description="Base URL for the OpenAI service",
    )
    rtsp_url: str = Field(
        default="rtsp://localhost:8554/top_camera",
        description="RTSP URL for the camera stream",
    )
    prompt: str = Field(
        default="What is the most interesting aspect in this series of images?",
        description="Prompt for the VLM",
    )
    fps: int = Field(default=15, description="Frames per second to process")
    descriptor_for_LLM: str = Field(
        default="Vision", description="Descriptor for LLM context"
    )


class VLMOpenAIRTSP(BaseVLMFuserInput[VLMOpenAIRTSPConfig]):
    """VLM input handler using OpenAI-compatible API with RTSP camera stream."""

    def __init__(self, config: VLMOpenAIRTSPConfig):
        super().__init__(config)

        api_key = self.config.api_key
        if api_key is None or api_key == "":
            raise ValueError("config file missing api_key")

        self.descriptor_for_LLM = self.config.descriptor_for_LLM

        self.vlm: VLMOpenAIRTSPProvider = VLMOpenAIRTSPProvider(
            base_url=self.config.base_url,
            api_key=api_key,
            rtsp_url=self.config.rtsp_url,
            prompt=self.config.prompt,
            fps=self.config.fps,
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

    def _handle_vlm_message(self, raw_message: ChatCompletion) -> None:
        """Extract content from OpenAI ChatCompletion response."""
        logging.info(f"VLM OpenAI received message: {raw_message}")
        content = raw_message.choices[0].message.content
        if content is not None:
            self.message_buffer.put(content)
