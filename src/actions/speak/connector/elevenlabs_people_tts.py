import logging
import re
from typing import Optional

from pydantic import Field

from actions.speak.connector.base_elevenlabs_tts import (
    BaseElevenLabsTTSConfig,
    BaseElevenLabsTTSConnector,
)
from actions.speak.interface import SpeakInput


class SpeakElevenLabsTTSConfig(BaseElevenLabsTTSConfig):
    """
    Configuration for ElevenLabs TTS connector with people-specific voice support.

    Parameters
    ----------
    voice_ids : Optional[dict[str, str]]
        Optional mapping of voice names to ElevenLabs voice IDs.
    """

    voice_ids: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional mapping of voice names to ElevenLabs voice IDs",
    )


class SpeakElevenLabsTTSConnector(BaseElevenLabsTTSConnector):
    """
    A "Speak" connector that uses the ElevenLabs TTS Provider to perform Text-to-Speech.
    This connector supports per-person voice selection based on FacePresenceInput.
    """

    config: SpeakElevenLabsTTSConfig  # type: ignore

    def _get_voice_id(self, output_interface: SpeakInput) -> str:
        """
        Get the voice ID to use for the current TTS request.
        Checks for people names from FacePresenceInput and maps them to voice IDs.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.

        Returns
        -------
        str
            The voice ID to use for TTS (either mapped from people name or default).
        """
        # Get people name from input if available
        people_name = None
        if self.io_provider.get_input("FacePresence") is not None:
            face_presence_input = self.io_provider.get_input("FacePresence")
            if (
                face_presence_input
                and face_presence_input.input
                and face_presence_input.tick == self.io_provider.tick_counter
            ):
                # Extract closest person name from camera view message
                # Format: "In Camera View: 1 known (boyuan). Closest: boyuan."
                # or "In Camera View: 2 known (shicai and lifan) and 2 unknown faces. Closest: shicai."
                input_text = face_presence_input.input
                match = re.search(r"Closest:\s*(\w+)", input_text)
                if match:
                    people_name = match.group(1).strip()
                    if people_name.lower() == "unknown":
                        people_name = None

        # Search people name in voice_ids mapping if available
        voice_ids = self.config.voice_ids
        if voice_ids and people_name and people_name in voice_ids:
            voice_id = voice_ids[people_name]
            logging.info(f"Found voice_id '{voice_id}' for people '{people_name}' in voice_ids mapping")
            return voice_id

        return self.config.voice_id
