import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class ElevenLabsTTSConfig(BackgroundConfig):
    """
    Configuration for Eleven Labs TTS Background.

    Parameters
    ----------
    api_key : Optional[str]
        OM API key.
    elevenlabs_api_key : Optional[str]
        Eleven Labs API key.
    voice_id : str
        Voice ID for TTS.
    model_id : str
        Model ID for TTS.
    output_format : str
        Output audio format.
    """

    api_key: Optional[str] = Field(default=None, description="OM API key")
    elevenlabs_api_key: Optional[str] = Field(
        default=None, description="Eleven Labs API key"
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb", description="Voice ID for TTS"
    )
    model_id: str = Field(default="eleven_flash_v2_5", description="Model ID for TTS")
    output_format: str = Field(
        default="mp3_44100_128", description="Output audio format"
    )


class ElevenLabsTTS(Background[ElevenLabsTTSConfig]):
    """
    Eleven Labs TTS Background Plugin.
    """

    def __init__(self, config: ElevenLabsTTSConfig):
        super().__init__(config)

        # OM API key
        api_key = self.config.api_key

        # Eleven Labs TTS configuration
        elevenlabs_api_key = self.config.elevenlabs_api_key
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format

        # Initialize Eleven Labs TTS Provider
        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        self.tts.start()

        # Configure Eleven Labs TTS Provider to ensure settings are applied
        self.tts.configure(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        logging.info("Eleven Labs TTS Provider initialized in background")
