from typing import Optional

from pydantic import Field

from inputs.asr_provider_base import BaseASRFuserInput
from inputs.base import SensorConfig
from providers.asr_provider import ASRProvider


class RivaASRSensorConfig(SensorConfig):
    """Configuration for Riva ASR Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rate : int
        Sampling rate.
    chunk : int
        Chunk size.
    base_url : str
        Base URL for the ASR service.
    microphone_device_id : Optional[str]
        Microphone Device ID.
    microphone_name : Optional[str]
        Microphone Name.
    remote_input : bool
        Whether to use remote input.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rate: int = Field(default=48000, description="Sampling rate")
    chunk: int = Field(default=12144, description="Chunk size")
    base_url: str = Field(
        default="wss://api-asr.openmind.org", description="Base URL for the ASR service"
    )
    stream_base_url: Optional[str] = Field(default=None, description="Stream Base URL")
    microphone_device_id: Optional[int] = Field(
        default=None, description="Microphone Device ID"
    )
    microphone_name: Optional[str] = Field(default=None, description="Microphone Name")
    remote_input: bool = Field(default=False, description="Whether to use remote input")
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class RivaASRInput(BaseASRFuserInput[RivaASRSensorConfig]):
    """Riva Automatic Speech Recognition (ASR) input handler.

    This class manages the input stream from a Riva ASR service,
    buffering messages and providing text conversion capabilities.
    """

    def __init__(self, config: RivaASRSensorConfig):
        super().__init__(config)

        rate = self.config.rate
        chunk = self.config.chunk
        base_url = self.config.base_url
        microphone_device_id = self.config.microphone_device_id
        microphone_name = self.config.microphone_name
        remote_input = self.config.remote_input
        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRProvider = ASRProvider(
            rate=rate,
            chunk=chunk,
            ws_url=base_url,
            device_id=microphone_device_id,
            microphone_name=microphone_name,
            remote_input=remote_input,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)
