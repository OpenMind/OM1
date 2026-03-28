import logging
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.episodic_memory_provider import EpisodicMemoryProvider


class StartEpisodicHookContext(BaseModel):
    """Context for starting episodic memory hook."""

    mode: str = Field(default="default", description="The mode being entered")
    announce: bool = Field(default=False, description="Whether to announce via TTS")
    model_config = ConfigDict(extra="allow")


class StopEpisodicHookContext(BaseModel):
    """Context for stopping episodic memory hook."""

    mode: str = Field(default="default", description="The mode being exited")
    announce: bool = Field(default=False, description="Whether to announce via TTS")
    model_config = ConfigDict(extra="allow")


async def start_episodic_hook(context: Dict[str, Any]):
    """Hook called when a mode with episodic memory is entered."""
    ctx = StartEpisodicHookContext(**context)
    provider = EpisodicMemoryProvider()
    provider._current_mode = ctx.mode
    provider.load_mode(ctx.mode)
    logging.info(f"episodic_hook: start — loaded episodes for mode '{ctx.mode}'")
    if ctx.announce:
        try:
            tts = ElevenLabsTTSProvider()
            tts.add_pending_message("I remember our past interactions.")
        except Exception as e:
            logging.warning(f"episodic_hook: TTS announce failed: {e}")
    return {
        "status": "success",
        "message": f"Episodic memory loaded for mode '{ctx.mode}'",
    }


async def stop_episodic_hook(context: Dict[str, Any]):
    """Hook called when a mode with episodic memory is exited."""
    ctx = StopEpisodicHookContext(**context)
    provider = EpisodicMemoryProvider()
    try:
        provider.save_to_disk()
        logging.info(f"episodic_hook: stop — flushed episodes for mode '{ctx.mode}'")
    except Exception as e:
        logging.error(f"episodic_hook: flush failed: {e}")
        raise
    if ctx.announce:
        try:
            tts = ElevenLabsTTSProvider()
            tts.add_pending_message("I've saved our memories.")
        except Exception as e:
            logging.warning(f"episodic_hook: TTS announce failed: {e}")
    return {
        "status": "success",
        "message": f"Episodic memory flushed for mode '{ctx.mode}'",
    }
