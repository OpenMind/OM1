import logging
from typing import Optional

from pydantic import Field

from inputs.asr_provider_base import BaseASRFuserInput
from inputs.base import SensorConfig
from providers.asr_rtsp_provider import ASRRTSPProvider

LANGUAGE_CODE_MAP: dict = {
    "english": "en-US",
    "chinese": "cmn-Hans-CN",
    "german": "de-DE",
    "french": "fr-FR",
    "japanese": "ja-JP",
    "korean": "ko-KR",
    "spanish": "es-ES",
    "italian": "it-IT",
    "portuguese": "pt-BR",
    "russian": "ru-RU",
    "arabic": "ar-SA",
}


class GoogleASRRTSPSensorConfig(SensorConfig):
    """Configuration for Google ASR RTSP Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rtsp_url : str
        RTSP URL for the audio stream.
    rate : int
        Audio sampling rate.
    base_url : Optional[str]
        Base URL for the ASR service.
    language : str
        Language for speech recognition.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rtsp_url: str = Field(
        default="rtsp://localhost:8554/audio",
        description="RTSP URL for the audio stream",
    )
    rate: int = Field(default=16000, description="Audio sampling rate")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the ASR service"
    )
    language: str = Field(
        default="english", description="Language for speech recognition"
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class GoogleASRRTSPInput(BaseASRFuserInput[GoogleASRRTSPSensorConfig]):
    """Google ASR RTSP input handler for processing speech recognition
    from RTSP audio streams.
    """

    def __init__(self, config: GoogleASRRTSPSensorConfig):
        super().__init__(config)

        api_key = self.config.api_key
        rtsp_url = self.config.rtsp_url
        rate = self.config.rate
        base_url = (
            self.config.base_url
            or f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
        )

        language = self.config.language.strip().lower()

        if language not in LANGUAGE_CODE_MAP:
            logging.error(
                f"Language {language} not supported. Current supported languages are : {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to English"
            )
            language = "english"

        language_code = LANGUAGE_CODE_MAP.get(language, "en-US")
        logging.info(f"Using language code {language_code} for Google ASR")

        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRRTSPProvider = ASRRTSPProvider(
            rtsp_url=rtsp_url,
            rate=rate,
            ws_url=base_url,
            language_code=language_code,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)
