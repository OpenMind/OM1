import speech_recognition as sr
import threading
import json
import logging

class ASRProvider:
    def __init__(self, **kwargs):
        self.is_multi = kwargs.get("is_multi_lang", False)
        self.callback = None
        self.running = False
        self.recognizer = sr.Recognizer()
        self.langs = ["tr-TR", "en-US", "fr-FR", "es-ES"]
        logging.info(f"ASRProvider başlatıldı. Çoklu Dil: {self.is_multi}")

    def register_message_callback(self, callback):
        self.callback = callback
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
        message_callback : Optional[Callable]
            The callback function to process ASR results.
        """
        if message_callback is not None:
            self.ws_client.register_message_callback(message_callback)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        # Mikrofonun sunucuda hata vermemesi için hata kontrolü
        try:
            with sr.Microphone() as source:
                while self.running:
                    try:
                        audio = self.recognizer.listen(source, phrase_time_limit=5)
                        # Çok dilli tarama
                        for lang in self.langs:
                            try:
                                text = self.recognizer.recognize_google(audio, language=lang)
                                if text and self.callback:
                                    self.callback(json.dumps({"asr_reply": text}))
                                    break
                            except: continue
                    except Exception: continue
        except Exception as e:
            logging.error(f"ASR Donanım Hatası (Normaldir, VMI üzerinde mikrofon yok): {e}")

    def stop(self):
        self.running = False
