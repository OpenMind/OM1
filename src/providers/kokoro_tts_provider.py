import logging
from typing import Callable, Optional, Union

from om1_speech import AudioOutputLiveStream

from .singleton import singleton


@singleton
class KokoroTTSProvider:
    """
    Text-to-Speech Provider that manages an audio output stream.

    This class provides an interface to configure and manage a TTS service using
    the Kokoro TTS engine. It supports setting voice, model, output format, and
    interrupt capabilities. The provider uses an underlying AudioOutputStream to
    handle audio playback.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8880/v1",
        api_key: Optional[str] = None,
        voice_id: str = "af_bella",
        model_id: str = "kokoro",
        output_format: str = "pcm",
        rate: int = 24000,
        enable_tts_interrupt: bool = False,
    ):
        """
        Initialize the KokoroTTSProvider instance.

        Sets up the configuration for the Kokoro TTS service, including API keys,
        voice/model selection, output format, and interrupt settings. It initializes
        the underlying audio output stream.

        Parameters
        ----------
        url : str, optional
            The URL endpoint for the TTS service. Defaults to "http://127.0.0.1:8880/v1"
        api_key : str, optional
            The API key for the TTS service. If provided, it's used in the request
            headers as "x-api-key". Defaults to None.
        voice_id : str
            The ID/name of the voice to use for TTS synthesis. Defaults to "af_bella".
        model_id : str
            The ID/name of the model to use for TTS synthesis. Defaults to "kokoro".
        output_format : str
            The desired audio output format (e.g., pcm, wav). Defaults to "pcm".
        rate : int
            The audio sample rate in Hz. Defaults to 24000.
        enable_tts_interrupt : bool, optional
            If True, enables the ability to interrupt ongoing TTS playback when ASR
            detects new speech input. Defaults to False.
        """
        self.api_key = api_key

        # Initialize TTS provider
        self.running: bool = False
        self._audio_stream: AudioOutputLiveStream = AudioOutputLiveStream(
            url=url,
            tts_model=model_id,
            tts_voice=voice_id,
            response_format=output_format,
            rate=rate,
            api_key=api_key,
            enable_tts_interrupt=enable_tts_interrupt,
        )

        # Set TTS parameters
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._enable_tts_interrupt = enable_tts_interrupt

    def configure(
        self,
        url: str = "http://127.0.0.1:8880/v1",
        api_key: Optional[str] = None,
        voice_id: str = "af_bella",
        model_id: str = "kokoro",
        output_format: str = "pcm",
        rate: int = 24000,
        enable_tts_interrupt: bool = False,
    ):
        """
        Configure the TTS provider with given parameters.

        Parameters
        ----------
        url : str, optional
            The URL endpoint for the TTS service. Defaults to "http://127.0.0.1:8880/v1"
        api_key : str, optional
            The API key for the TTS service. If provided, it's used in the request
            headers as "x-api-key". Defaults to None.
        voice_id : str
            The ID/name of the voice to use for TTS synthesis. Defaults to "af_bella".
        model_id : str
            The ID/name of the model to use for TTS synthesis. Defaults to "kokoro".
        output_format : str
            The desired audio output format (e.g., pcm, wav). Defaults to "pcm".
        rate : int
            The audio sample rate in Hz. Defaults to 24000.
        enable_tts_interrupt : bool, optional
            If True, enables the ability to interrupt ongoing TTS playback when ASR
            detects new speech input. Defaults to False.
        """
        restart_needed = (
            url != self._audio_stream._url
            or api_key != self.api_key
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
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._enable_tts_interrupt = enable_tts_interrupt

        self._audio_stream: AudioOutputLiveStream = AudioOutputLiveStream(
            url=url,
            tts_model=model_id,
            tts_voice=voice_id,
            response_format=output_format,
            rate=rate,
            api_key=api_key,
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
        return {
            "text": text,
            "voice_id": self._voice_id,
            "model_id": self._model_id,
            "output_format": self._output_format,
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

        logging.info(f"Adding pending TTS message: {message}")
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
            logging.warning("Kokoro TTS provider is already running")
            return

        self.running = True
        self._audio_stream.start()

    def stop(self):
        """
        Stop the TTS provider and cleanup resources.
        """
        if not self.running:
            logging.warning("Kokoro TTS provider is not running")
            return

        self.running = False
        self._audio_stream.stop()
