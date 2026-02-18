import json
import logging
from typing import Dict

from pydantic import Field

from inputs.base import SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput
from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider


class VLMVilaZenohConfig(SensorConfig):
    """Configuration for VLM Vila Zenoh Sensor."""

    base_url: str = Field(
        default="wss://api-vila.openmind.org",
        description="Base URL for the VLM service",
    )
    topic: str = Field(
        default="rgb_image", description="Zenoh topic for receiving images"
    )
    decode_format: str = Field(
        default="H264", description='Image decode format (e.g., "H264")'
    )


class VLMVilaZenoh(BaseVLMFuserInput[VLMVilaZenohConfig]):
    """VLM input handler using Vila API with Zenoh image transport."""

    def __init__(self, config: VLMVilaZenohConfig):
        super().__init__(config)

        self.vlm: VLMVilaZenohProvider = VLMVilaZenohProvider(
            ws_url=self.config.base_url,
            topic=self.config.topic,
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
