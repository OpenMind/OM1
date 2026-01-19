import logging
from typing import Callable, Optional

from om1_speech import AudioOutputStream

from .singleton import singleton


@singleton
class RivaTTSProvider:
    """
    Text-to-Speech Provider that manages an audio output stream.

    A singleton class that handles text-to-speech conversion and audio output
    through a dedicated thread.

    Parameters
    ----------
    url : str
        The URL endpoint for the TTS service
    api_key : str, optional
        The API key for the TTS service (default is None)
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the TTS provider with given URL and API key.

        Parameters
        ----------
        url : str
            The URL endpoint for the TTS service.
        api_key : str, optional
            The API key for the TTS service. If provided, it's used in the
            request headers. Defaults to None.
        """
        self.running: bool = False
        self._audio_stream: AudioOutputStream = AudioOutputStream(
            url=url,
            headers={"x-api-key": api_key} if api_key else None,
        )

    def register_tts_state_callback(self, tts_state_callback: Optional[Callable]):
        """
        Register a callback for TTS state changes.

        Parameters
        ----------
        tts_state_callback : Optional[callable]
            The callback function to receive TTS state changes.
        """
        if tts_state_callback is not None:
            self._audio_stream.set_tts_state_callback(tts_state_callback)

    def add_pending_message(self, text: str):
        """
        Add text to the pending queue for TTS processing.

        Parameters
        ----------
        text : str
            Text to be converted to speech
        """
        logging.info(f"audio_stream: {text}")
        self._audio_stream.add_request({"text": text})

    def start(self):
        """
        Start the TTS provider and its audio stream.
        """
        if self.running:
            logging.warning("Riva TTS provider is already running")
            return

        self.running = True
        self._audio_stream.start()

    def stop(self):
        """
        Stop the TTS provider and cleanup resources.
        """
        self.running = False
        self._audio_stream.stop()
