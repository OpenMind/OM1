from typing import Optional

from pydantic import Field

from inputs.asr_provider_base import BaseASRFuserInput
from inputs.base import SensorConfig
from providers.asr_rtsp_provider import ASRRTSPProvider


class RivaASRRTSPSensorConfig(SensorConfig):
    """Configuration for Riva ASR Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rtsp_url : str
        RTSP URL for the audio stream. Default is "rtsp://localhost:8554/audio".
    rate : int
        Sampling rate. Default is 16000.
    base_url : str
        Base URL for the ASR service. Default is "wss://api-asr.openmind.org".
    enable_tts_interrupt : bool
        Enable TTS interrupt when ASR detects speech during playback.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rtsp_url: str = Field(
        default="rtsp://localhost:8554/audio",
        description="RTSP URL for the audio stream",
    )
    rate: int = Field(default=16000, description="Sampling rate")
    base_url: str = Field(
        default="wss://api-asr.openmind.org", description="Base URL for the ASR service"
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class RivaASRRTSPInput(BaseASRFuserInput[RivaASRRTSPSensorConfig]):
    """Riva ASR RTSP input handler for processing speech recognition
    from RTSP audio streams.
    """

    def __init__(self, config: RivaASRRTSPSensorConfig):
        super().__init__(config)

        rtsp_url = self.config.rtsp_url
        rate = self.config.rate
        base_url = self.config.base_url
        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRRTSPProvider = ASRRTSPProvider(
            rtsp_url=rtsp_url,
            rate=rate,
            ws_url=base_url,
            language_code="en-US",
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)
