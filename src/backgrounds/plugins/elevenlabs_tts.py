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
    Background task for text-to-speech synthesis using Eleven Labs TTS service.

    This background task initializes and manages an ElevenLabsTTSProvider instance
    that handles text-to-speech conversion through the Eleven Labs API. The provider
    processes text input and generates high-quality audio output using configured
    voice and model settings.

    The Eleven Labs TTS service provides natural-sounding speech synthesis with
    various voice options and model configurations, enabling robots to communicate
    verbally with users in a more engaging and human-like manner.
    """

    def __init__(self, config: ElevenLabsTTSConfig):
        """
        Initialize Eleven Labs TTS background task with configuration.

        Parameters
        ----------
        config : ElevenLabsTTSConfig
            Configuration object containing API keys, voice settings, model selection,
            and output format preferences. The configuration includes:
            - OM API key for OpenMind service authentication
            - Eleven Labs API key for TTS service access
            - Voice ID specifying the voice to use for synthesis
            - Model ID determining the TTS model (e.g., "eleven_flash_v2_5")
            - Output format defining audio encoding (e.g., "mp3_44100_128")
        """
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
