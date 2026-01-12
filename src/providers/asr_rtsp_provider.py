import logging
from typing import Callable, Optional

from om1_speech import AudioRTSPInputStream
from om1_utils import ws

from .singleton import singleton


@singleton
class ASRRTSPProvider:
    """
    Audio Speech Recognition Provider that handles RTSP audio streaming and websocket communication.

    This class implements a singleton pattern to manage audio input streaming and websocket
    communication for speech recognition services. It runs in a separate thread to handle
    continuous audio processing.
    """

    def __init__(
        self,
        ws_url: str,
        rtsp_url: str = "rtsp://localhost:8554/audio",
        rate: int = 48000,
        chunk: Optional[int] = None,
        language_code: Optional[str] = None,
        enable_tts_interrupt: bool = False,
    ):
        """
        Initialize the ASR Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the ASR service connection.
        rtsp_url : str
            The RTSP URL for the audio stream; defaults to "rtsp://localhost:8554/audio"
        rate : int
            The audio sample rate for the audio stream; used the system default if None
        chunk : int
            The audio chunk size for the audio stream; used the 200ms default if None
        language_code : str
            The language code for language in the audio stream; used the en-US default if None
        enable_tts_interrupt : bool
            If True, enables TTS interrupt.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.audio_stream: AudioRTSPInputStream = AudioRTSPInputStream(
            rtsp_url=rtsp_url,
            rate=rate,
            chunk=chunk,
            audio_data_callback=self.ws_client.send_message,
            language_code=language_code,
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
            logging.warning("ASR RTSP provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.audio_stream.start()

        logging.info("ASR RTSP provider started")

    def stop(self):
        """
        Stop the ASR provider.

        Stops the audio stream and websocket clients, and sets the running state to False.
        """
        if not self.running:
            logging.warning("ASR RTSP provider is not running")
            return

        self.running = False
        self.audio_stream.stop()
        self.ws_client.stop()
