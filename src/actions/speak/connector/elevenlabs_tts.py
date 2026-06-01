from actions.speak.connector.base_elevenlabs_tts import (
    BaseElevenLabsTTSConfig,
    BaseElevenLabsTTSConnector,
)
from actions.speak.interface import SpeakInput


class SpeakElevenLabsTTSConfig(BaseElevenLabsTTSConfig):
    """
    Configuration for ElevenLabs TTS connector.
    Inherits all configuration from BaseElevenLabsTTSConfig.
    """

    pass


class SpeakElevenLabsTTSConnector(BaseElevenLabsTTSConnector):
    """
    A "Speak" connector that uses the ElevenLabs TTS Provider to perform Text-to-Speech.
    This connector uses a fixed voice ID for all TTS requests.
    """

    def _get_voice_id(self, output_interface: SpeakInput) -> str:
        """
        Get the voice ID to use for the current TTS request.
        Always returns the configured voice_id from config.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.

        Returns
        -------
        str
            The configured voice ID.
        """
        return self.config.voice_id
