import json
import logging
from typing import Dict, Optional

from pydantic import Field

from inputs.base import SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput
from providers.vlm_vila_provider import VLMVilaProvider


class VLMVilaConfig(SensorConfig):
    """Configuration for VLM Vila Sensor."""

    api_key: Optional[str] = Field(default=None, description="API Key")
    base_url: str = Field(
        default="wss://api-vila.openmind.com",
        description="Base URL for the VLM service",
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    camera_index: int = Field(default=0, description="Index of the camera device")


class VLMVila(BaseVLMFuserInput[VLMVilaConfig]):
    """VLM input handler using Vila API with local camera."""

    def __init__(self, config: VLMVilaConfig):
        super().__init__(config)

        api_key = self.config.api_key
        stream_base_url = (
            self.config.stream_base_url
            or f"wss://api.openmind.com/api/core/teleops/stream/video?api_key={api_key}"
        )

        self.vlm: VLMVilaProvider = VLMVilaProvider(
            ws_url=self.config.base_url,
            stream_url=stream_base_url,
            camera_index=self.config.camera_index,
        )
        self.vlm.start()
        self.vlm.register_message_callback(self._handle_vlm_message)

    def _handle_vlm_message(self, raw_message: str) -> None:
        """Extract vlm_reply from JSON message."""
        try:
            json_message: Dict = json.loads(raw_message)
            if "vlm_reply" in json_message:
                vlm_reply = json_message["vlm_reply"]
                self.message_buffer.put(vlm_reply)
                logging.info("Detected VLM message: %s", vlm_reply)
        except json.JSONDecodeError:
            pass
