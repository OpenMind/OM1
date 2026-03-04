import logging
from typing import Any

from pydantic import Field

from actions.base import ActionConfig
from actions.greeting_conversation.connector.base_greeting_conversation import (
    BaseGreetingConversationConnector,
)
from providers.kokoro_tts_provider import KokoroTTSProvider


class SpeakKokoroTTSConfig(ActionConfig):
    """
    Configuration for Kokoro TTS connector.

    Parameters
    ----------
    base_url : str
        Base URL for Kokoro TTS API.
    voice_id : str
        Kokoro voice ID.
    model_id : str
        Kokoro model ID.
    output_format : str
        Kokoro output format.
    rate : int
        Audio sample rate in Hz.
    enable_tts_interrupt : bool
        Enable TTS interrupt when ASR detects speech during playback.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    base_url: str = Field(
        default="http://127.0.0.1:8880/v1",
        description="Base URL for Kokoro TTS API",
    )
    voice_id: str = Field(
        default="af_bella",
        description="Kokoro voice ID",
    )
    model_id: str = Field(
        default="kokoro",
        description="Kokoro model ID",
    )
    output_format: str = Field(
        default="pcm",
        description="Kokoro output format",
    )
    rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt when ASR detects speech during playback",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )


class GreetingConversationConnector(
    BaseGreetingConversationConnector[SpeakKokoroTTSConfig]
):
    """
    Connector that manages greeting conversations using Kokoro TTS.
    """

    def create_tts_provider(self) -> Any:
        """
        Create and return the Kokoro TTS provider.

        Returns
        -------
        KokoroTTSProvider
            The instantiated Kokoro TTS provider.
        """
        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Kokoro TTS configuration
        base_url = self.config.base_url
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format
        rate = self.config.rate
        enable_tts_interrupt = self.config.enable_tts_interrupt

        logging.info("Creating Kokoro TTS provider")
        return KokoroTTSProvider(
            url=base_url,
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            rate=rate,
            enable_tts_interrupt=enable_tts_interrupt,
        )
