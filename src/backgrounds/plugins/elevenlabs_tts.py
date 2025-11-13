import logging

from backgrounds.base import Background, BackgroundConfig
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class ElevenLabsTTS(Background):
    """
    Eleven Labs TTS Background Plugin.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Eleven Labs TTS configuration
        elevenlabs_api_key = getattr(self.config, "elevenlabs_api_key", None)
        voice_id = getattr(self.config, "voice_id", "JBFqnCBsd6RMkjVDRZzb")
        model_id = getattr(self.config, "model_id", "eleven_flash_v2_5")
        output_format = getattr(self.config, "output_format", "mp3_44100_128")

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
        logging.info("Eleven Labs TTS Provider initialized in background")
