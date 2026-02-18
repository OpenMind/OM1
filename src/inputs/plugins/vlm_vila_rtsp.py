import json
import logging
from typing import Dict

from pydantic import Field

from inputs.base import SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput
from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider


class VLMVilaRTSPConfig(SensorConfig):
    """Configuration for VLM Vila RTSP Sensor."""

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


class VLMVilaRTSP(BaseVLMFuserInput[VLMVilaRTSPConfig]):
    """VLM input handler using Vila API with RTSP camera stream."""

    def __init__(self, config: VLMVilaRTSPConfig):
        super().__init__(config)

        self.vlm: VLMVilaRTSPProvider = VLMVilaRTSPProvider(
            ws_url=self.config.base_url,
            rtsp_url=self.config.rtsp_url,
            decode_format=self.config.decode_format,
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
