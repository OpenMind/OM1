import logging
from typing import Optional

from pydantic import Field

from inputs.asr_provider_base import BaseASRFuserInput
from inputs.base import SensorConfig
from providers.asr_provider import ASRProvider

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


class GoogleASRSensorConfig(SensorConfig):
    """Configuration for Google ASR Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    rate : int
        Sampling rate.
    chunk : int
        Chunk size.
    base_url : Optional[str]
        Base URL for the ASR service.
    microphone_device_id : Optional[str]
        Microphone Device ID.
    microphone_name : Optional[str]
        Microphone Name.
    language : str
        Language for speech recognition.
    remote_input : bool
        Whether to use remote input.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    rate: int = Field(default=48000, description="Sampling rate")
    chunk: int = Field(default=12144, description="Chunk size")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the ASR service"
    )
    microphone_device_id: Optional[int] = Field(
        default=None, description="Microphone Device ID"
    )
    microphone_name: Optional[str] = Field(default=None, description="Microphone Name")
    language: str = Field(
        default="english", description="Language for speech recognition"
    )
    remote_input: bool = Field(default=False, description="Whether to use remote input")
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt (does not mute mic during TTS playback)",
    )


class GoogleASRInput(BaseASRFuserInput[GoogleASRSensorConfig]):
    """Google Automatic Speech Recognition (ASR) input handler.

    This class manages the input stream from a Google ASR service,
    buffering messages and providing text conversion capabilities.
    """

    def __init__(self, config: GoogleASRSensorConfig):
        super().__init__(config)

        api_key = self.config.api_key
        rate = self.config.rate
        chunk = self.config.chunk
        base_url = (
            self.config.base_url
            or f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}"
        )
        microphone_device_id = self.config.microphone_device_id
        microphone_name = self.config.microphone_name

        language = self.config.language.strip().lower()

        if language not in LANGUAGE_CODE_MAP:
            logging.error(
                f"Language {language} not supported. Current supported languages are : {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to English"
            )
            language = "english"

        language_code = LANGUAGE_CODE_MAP.get(language, "en-US")
        logging.info(f"Using language code {language_code} for Google ASR")

        remote_input = self.config.remote_input
        enable_tts_interrupt = self.config.enable_tts_interrupt

        self.asr: ASRProvider = ASRProvider(
            rate=rate,
            chunk=chunk,
            ws_url=base_url,
            device_id=microphone_device_id,
            microphone_name=microphone_name,
            language_code=language_code,
            remote_input=remote_input,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.asr.start()
        self.asr.register_message_callback(self._handle_asr_message)
