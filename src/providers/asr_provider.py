import logging
from typing import Callable, Optional

from om1_speech import AudioInputStream
from om1_utils import ws

from .singleton import singleton


@singleton
class ASRProvider:
    """
    Audio Speech Recognition Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage audio input streaming and websocket
    communication for speech recognition services. It runs in a separate thread to handle
    continuous audio processing.
    """

    def __init__(
        self,
        ws_url: str,
        stream_url: Optional[str] = None,
        device_id: Optional[int] = None,
        microphone_name: Optional[str] = None,
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        language_code: Optional[str] = None,
        remote_input: bool = False,
        enable_tts_interrupt: bool = False,
    ):
        """
        Initialize the ASR Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the ASR service connection.
        device_id : int
            The device ID of the chosen microphone; used the system default if None
        microphone_name : str
            The name of the microphone to use for audio input
        rate : int
            The audio sample rate for the audio stream; used the system default if None
        chunk : int
            The audio chunk size for the audio stream; used the 200ms default if None
        language_code : str
            The language code for language in the audio stream; used the en-US default if None
        remote_input : bool
            If True, the audio input is processed remotely; defaults to False.
        enable_tts_interrupt : bool
            If True, enables TTS interrupt.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.audio_stream: AudioInputStream = AudioInputStream(
            rate=rate,
            chunk=chunk,
            device=device_id,
            device_name=microphone_name,  # type: ignore
            audio_data_callback=self.ws_client.send_message,
            language_code=language_code,
            remote_input=remote_input,
            enable_tts_interrupt=enable_tts_interrupt,
        )

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing ASR results.

        Parameters
        ----------
        callback : Optional[Callable])
            The callback function to process ASR results.
        """
        if message_callback is not None:
            self.ws_client.register_message_callback(message_callback)

    def start(self):
        """
        Start the ASR provider.

        Initializes and starts the websocket client, audio stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("ASR provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.audio_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.audio_stream.register_audio_data_callback(
                self.stream_ws_client.send_message
            )
            # Register the audio stream to fill the buffer for remote input
            if self.audio_stream.remote_input:
                self.stream_ws_client.register_message_callback(
                    self.audio_stream.fill_buffer_remote
                )

        logging.info("ASR provider started")

    def stop(self):
        """
        Stop the ASR provider.

        Stops the audio stream and websocket clients, and sets the running state to False.
        """
        self.running = False
        self.audio_stream.stop()
        self.ws_client.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()
