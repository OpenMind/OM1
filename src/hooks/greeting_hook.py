import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.face_presence_provider import FacePresenceProvider
from providers.greeting_conversation_state_provider import (
    GreetingConversationStateMachineProvider,
)
from providers.kokoro_tts_provider import KokoroTTSProvider
from providers.riva_tts_provider import RivaTTSProvider


class GreetingHookContext(BaseModel):
    """
    Configuration for greeting hooks.

    Parameters
    ----------
    message : str
        The message to log or announce. Supports {variable} formatting.
    tts_provider : str
        The TTS provider to use ('elevenlabs', 'kokoro', 'riva'). Defaults to 'elevenlabs'.
    base_url : Optional[str]
        The URL endpoint for the TTS service. Provider-specific defaults apply.
    api_key : Optional[str]
        OpenMind API key for TTS service.
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key (only for 'elevenlabs' provider).
    voice_id : str
        Voice ID for the TTS provider.
    model_id : str
        Model ID for the TTS provider.
    output_format : str
        Audio output format.
    rate : Optional[int]
        Audio sample rate in Hz.
    enable_tts_interrupt : bool
        Enable TTS interrupt capability.
    robot_name : Optional[str]
        Name of the robot to use in greetings.
    custom_message : Optional[str]
        Custom message to use instead of the default greeting.
    """

    message: str = Field(
        default="",
        description="The message to log or announce. Supports {variable} formatting.",
    )
    tts_provider: str = Field(
        default="elevenlabs",
        description="The TTS provider to use ('elevenlabs', 'kokoro', 'riva')",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="The URL endpoint for the TTS service. Provider-specific defaults apply.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="OpenMind API key for TTS service",
    )
    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key (only for 'elevenlabs' provider)",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for the TTS provider",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Model ID for the TTS provider",
    )
    output_format: Optional[str] = Field(
        default=None,
        description="Audio output format",
    )
    rate: Optional[int] = Field(
        default=None,
        description="Audio sample rate in Hz",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt capability",
    )

    robot_name: Optional[str] = Field(
        default=None,
        description="Name of the robot to use in greetings",
    )
    custom_message: Optional[str] = Field(
        default=None,
        description="Custom message to use instead of the default greeting",
    )

    model_config = ConfigDict(extra="allow")


async def greeting_start_hook(context: Dict[str, Any]):
    """
    Hook to handle the start of a greeting conversation.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing relevant information for the hook.
    """
    ctx = GreetingHookContext(**context)

    tts_provider = ctx.tts_provider.lower()
    provider = None

    try:
        if tts_provider == "elevenlabs":
            provider = ElevenLabsTTSProvider(
                url=ctx.base_url or "https://api.openmind.com/api/core/elevenlabs/tts",
                api_key=ctx.api_key,
                elevenlabs_api_key=ctx.elevenlabs_api_key,
                voice_id=ctx.voice_id or "JBFqnCBsd6RMkjVDRZzb",
                model_id=ctx.model_id or "eleven_flash_v2_5",
                output_format=ctx.output_format or "pcm_16000",
                rate=ctx.rate or 16000,
                enable_tts_interrupt=ctx.enable_tts_interrupt,
            )
        elif tts_provider == "kokoro":
            provider = KokoroTTSProvider(
                url=ctx.base_url or "http://127.0.0.1:8880/v1",
                api_key=ctx.api_key,
                voice_id=ctx.voice_id or "af_bella",
                model_id=ctx.model_id or "kokoro",
                output_format=ctx.output_format or "pcm",
                rate=ctx.rate or 24000,
                enable_tts_interrupt=ctx.enable_tts_interrupt,
            )
        elif tts_provider == "riva":
            provider = RivaTTSProvider(
                url=ctx.base_url or "http://127.0.0.1:50051",
                api_key=ctx.api_key,
            )
        else:
            raise ValueError(
                f"Unsupported TTS provider: {tts_provider}. " f"Supported providers are: elevenlabs, kokoro, riva"
            )

        provider.start()

        robot_intro = f"I'm {ctx.robot_name}. " if ctx.robot_name else ""
        help_message = ctx.custom_message if ctx.custom_message else "How can I help you today?"

        face_presence_provider = FacePresenceProvider()
        face_snapshot_status = await face_presence_provider.fetch_snapshot()
        if face_snapshot_status is not None:
            closet_name = face_snapshot_status.closest_name
            if closet_name.lower() != "unknown":
                provider.add_pending_message(f"Hello {closet_name}! {robot_intro}Nice to see you. {help_message}")
            else:
                provider.add_pending_message(f"Hello! {robot_intro}{help_message}")
        else:
            provider.add_pending_message(f"Hello! {robot_intro}{help_message}")

        logging.info("Greeting start hook executed successfully.")

        return True
    except Exception:
        logging.exception("Error in greeting_start_hook")
        return False


async def greeting_end_hook(context: Dict[str, Any]):
    """
    Hook to handle the end of a greeting conversation.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing relevant information for the hook.
    """
    ctx = GreetingHookContext(**context)

    tts_provider = ctx.tts_provider.lower()
    provider = None

    try:
        if tts_provider == "elevenlabs":
            provider = ElevenLabsTTSProvider(
                url=ctx.base_url or "https://api.openmind.com/api/core/elevenlabs/tts",
                api_key=ctx.api_key,
                elevenlabs_api_key=ctx.elevenlabs_api_key,
                voice_id=ctx.voice_id or "JBFqnCBsd6RMkjVDRZzb",
                model_id=ctx.model_id or "eleven_flash_v2_5",
                output_format=ctx.output_format or "pcm_16000",
                rate=ctx.rate or 16000,
                enable_tts_interrupt=ctx.enable_tts_interrupt,
            )
        elif tts_provider == "kokoro":
            provider = KokoroTTSProvider(
                url=ctx.base_url or "http://127.0.0.1:8880/v1",
                api_key=ctx.api_key,
                voice_id=ctx.voice_id or "af_bella",
                model_id=ctx.model_id or "kokoro",
                output_format=ctx.output_format or "pcm",
                rate=ctx.rate or 24000,
                enable_tts_interrupt=ctx.enable_tts_interrupt,
            )
        elif tts_provider == "riva":
            provider = RivaTTSProvider(
                url=ctx.base_url or "http://127.0.0.1:50051",
                api_key=ctx.api_key,
            )
        else:
            raise ValueError(
                f"Unsupported TTS provider: {tts_provider}. " f"Supported providers are: elevenlabs, kokoro, riva"
            )

        provider.start()

        greeting_state_provider = GreetingConversationStateMachineProvider()
        if greeting_state_provider.turn_count >= greeting_state_provider.max_turn_count:
            logging.info("Greeting conversation ended due to maximum turn count.")
            provider.add_pending_message("Thank you for chatting with me today. I hope you enjoy the rest of your day.")
        elif greeting_state_provider.turn_count > 0:
            provider.add_pending_message(
                "It was nice talking with you! If you have any more questions, come chat with me again!"
            )
        else:
            provider.add_pending_message(
                "It was great meeting you! If you want to chat later, just come back and say hi!"
            )

        return True

    except Exception:
        logging.exception("Error in greeting_end_hook")
        return False
