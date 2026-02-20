import logging
from typing import Any, Optional

from pydantic import Field

from actions.base import ActionConfig
from actions.greeting_conversation.connector.base_greeting_conversation import (
    BaseGreetingConversationConnector,
)
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class SpeakElevenLabsTTSConfig(ActionConfig):
    """
    Configuration for ElevenLabs TTS connector.

    Parameters
    ----------
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key.
    voice_id : str
        ElevenLabs voice ID.
    model_id : str
        ElevenLabs model ID.
    output_format : str
        ElevenLabs output format.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key",
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="ElevenLabs voice ID",
    )
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="ElevenLabs model ID",
    )
    output_format: str = Field(
        default="pcm_16000",
        description="ElevenLabs output format",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )


class GreetingConversationConnector(
    BaseGreetingConversationConnector[SpeakElevenLabsTTSConfig]
):
    """
    Connector that manages greeting conversations using ElevenLabs TTS.
    """

    def create_tts_provider(self) -> Any:
        """
        Create and return the ElevenLabs TTS provider.

        Returns
        -------
        ElevenLabsTTSProvider
            The instantiated ElevenLabs TTS provider.
        """
        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Eleven Labs TTS configuration
        elevenlabs_api_key = self.config.elevenlabs_api_key
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format

        logging.info("Creating ElevenLabs TTS provider")
        return ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
