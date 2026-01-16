import logging
from typing import Callable, Optional, Union

from om1_speech import AudioOutputStream

from .singleton import singleton


@singleton
class ElevenLabsTTSProvider:
    """
    Text-to-Speech Provider that manages an audio output stream.

    A singleton class that handles text-to-speech conversion and audio output
    through a dedicated thread.

    Parameters
    ----------
    url : str
        The URL endpoint for the TTS service. (Default is https://api.openmind.org/api/core/elevenlabs/tts)
    api_key : str
        The API key for the TTS service
    voice_id : str, optional
        The name of the voice for Eleven Labs TTS service (default is JBFqnCBsd6RMkjVDRZzb)
    model_id : str, optional
        The name of the model for Eleven Labs TTS service (default is eleven_multilingual
    output_format : str, optional
        The output format for the audio stream (default is mp3_44100_128)
    """

    def __init__(
        self,
        url: str = "https://api.openmind.org/api/core/elevenlabs/tts",
        api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb",
        model_id: Optional[str] = "eleven_flash_v2_5",
        output_format: Optional[str] = "mp3_44100_128",
        enable_tts_interrupt: bool = False,
    ):
        """
        Initialize the TTS provider with given URL.
        """
        self.api_key = api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self._enable_tts_interrupt = enable_tts_interrupt

        # Initialize TTS provider
        self.running: bool = False
        self._audio_stream: AudioOutputStream = AudioOutputStream(
            url=url,
            headers={"x-api-key": api_key} if api_key else None,
            enable_tts_interrupt=enable_tts_interrupt,
        )

        # Set Eleven Labs TTS parameters
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format

    def configure(
        self,
        url: str = "https://api.openmind.org/api/core/elevenlabs/tts",
        api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb",
        model_id: Optional[str] = "eleven_flash_v2_5",
        output_format: Optional[str] = "mp3_44100_128",
        enable_tts_interrupt: bool = False,
    ):
        """
        Configure the TTS provider with given parameters.

        Parameters
        ----------
        url : str
            The URL endpoint for the TTS service.
        api_key : str
            The API key for the TTS service.
        voice_id : str, optional
            The name of the voice for Eleven Labs TTS service.
        model_id : str, optional
            The name of the model for Eleven Labs TTS service.
        output_format : str, optional
            The output format for the audio stream.
        enable_tts_interrupt : bool
            If True, enables TTS interrupt when ASR detects speech.
        """
        restart_needed = (
            url != self._audio_stream._url
            or api_key != self.api_key
            or elevenlabs_api_key != self.elevenlabs_api_key
            or voice_id != self._voice_id
            or model_id != self._model_id
            or output_format != self._output_format
            or enable_tts_interrupt != self._enable_tts_interrupt
        )

        if not restart_needed:
            return

        if self.running:
            self.stop()

        self.api_key = api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._enable_tts_interrupt = enable_tts_interrupt

        self._audio_stream: AudioOutputStream = AudioOutputStream(
            url=url,
            headers={"x-api-key": api_key} if api_key else None,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self._audio_stream.start()

    def register_tts_state_callback(self, tts_state_callback: Optional[Callable]):
        """
        Register a callback for TTS state changes.

        Parameters
        ----------
        tts_state_callback : Optional[Callable]
            The callback function to receive TTS state changes.
        """
        if tts_state_callback is not None:
            self._audio_stream.set_tts_state_callback(tts_state_callback)

    def create_pending_message(self, text: str) -> dict:
        """
        Create a pending message for TTS processing.

        Parameters
        ----------
        text : str
            Text to be converted to speech

        Returns
        -------
        dict
            A dictionary containing the TTS request parameters.
        """
        logging.info(f"audio_stream: {text}")
        elevenlabs_api_key = (
            {"elevenlabs_api_key": self.elevenlabs_api_key}
            if self.elevenlabs_api_key
            else {}
        )
        return {
            "text": text,
            "voice_id": self._voice_id,
            "model_id": self._model_id,
            "output_format": self._output_format,
            **elevenlabs_api_key,
        }

    def add_pending_message(self, message: Union[str, dict]):
        """
        Add a pending message to the TTS provider.

        Parameters
        ----------
        message : Union[str, dict]
            The message to be added, typically containing text and TTS parameters.
        """
        if not self.running:
            logging.warning(
                "TTS provider is not running. Call start() before adding messages."
            )
            return

        if isinstance(message, str):
            message = self.create_pending_message(message)
        self._audio_stream.add_request(message)

    def get_pending_message_count(self) -> int:
        """
        Get the count of pending messages in the TTS provider.

        Returns
        -------
        int
            The number of pending messages.
        """
        return self._audio_stream._pending_requests.qsize()

    def start(self):
        """
        Start the TTS provider and its audio stream.
        """
        if self.running:
            logging.warning("Eleven Labs TTS provider is already running")
            return

        self.running = True
        self._audio_stream.start()

    def stop(self):
        """
        Stop the TTS provider and cleanup resources.
        """
        if not self.running:
            logging.warning("Eleven Labs TTS provider is not running")
            return

        self.running = False
        self._audio_stream.stop()
