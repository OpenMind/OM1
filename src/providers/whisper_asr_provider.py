import json
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from .singleton import singleton

SUPPORTED_MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3", "turbo")


@singleton
class WhisperASRProvider:
    """
    Local Whisper ASR Provider using faster-whisper for speech recognition.

    This class implements a singleton pattern to manage local audio transcription
    using OpenAI's Whisper model via the faster-whisper CTranslate2 backend.
    Audio is captured directly from the microphone via PyAudio and transcribed
    locally without any cloud API dependency.

    Parameters
    ----------
    model_size : str
        Whisper model size. One of: tiny, base, small, medium, large-v3, turbo.
    device : str
        Compute device: "cuda", "cpu", or "auto".
    compute_type : str
        Compute precision: "float16", "int8", "float32", or "auto".
    language : Optional[str]
        Language code (e.g., "en", "tr", "ja"). None for auto-detection.
    device_id : Optional[int]
        Microphone device ID.
    microphone_name : Optional[str]
        Microphone device name.
    rate : Optional[int]
        Audio sample rate.
    chunk : Optional[int]
        Audio chunk size.
    enable_tts_interrupt : bool
        If True, enables TTS interrupt.
    """

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
        device_id: Optional[int] = None,
        microphone_name: Optional[str] = None,
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        enable_tts_interrupt: bool = False,
    ):
        if model_size not in SUPPORTED_MODEL_SIZES:
            logging.warning(
                f"Unsupported model size '{model_size}'. "
                f"Supported: {SUPPORTED_MODEL_SIZES}. Defaulting to 'turbo'."
            )
            model_size = "turbo"

        self.running: bool = False
        self.language: Optional[str] = language
        self._message_callback: Optional[Callable] = None

        # Audio buffer for accumulating chunks before transcription
        self._audio_buffer: list[bytes] = []
        self._buffer_lock = threading.Lock()

        # Silence detection parameters
        self._silence_threshold = 500  # RMS amplitude threshold
        self._silence_duration = 0.8  # Seconds of silence to trigger transcription
        self._silent_chunks = 0
        self._has_speech = False

        logging.info(
            f"Loading Whisper model '{model_size}' on device='{device}' "
            f"compute_type='{compute_type}'"
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info(f"Whisper model '{model_size}' loaded successfully")

        self._rate = rate or 16000
        self._chunk = chunk or 3200

        # Resolve microphone device
        self._pa = pyaudio.PyAudio()
        self._device_index = self._resolve_device(device_id, microphone_name)
        self._stream: Optional[pyaudio.Stream] = None
        self._audio_thread: Optional[threading.Thread] = None

        self._transcription_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._transcription_thread: Optional[threading.Thread] = None

    def _resolve_device(
        self, device_id: Optional[int], microphone_name: Optional[str]
    ) -> Optional[int]:
        """
        Resolve microphone device by ID or name.

        Parameters
        ----------
        device_id : Optional[int]
            Explicit device ID.
        microphone_name : Optional[str]
            Device name to search for.

        Returns
        -------
        Optional[int]
            Resolved device index, or None for system default.
        """
        if device_id is not None:
            return device_id

        if microphone_name:
            for i in range(self._pa.get_device_count()):
                info = self._pa.get_device_info_by_index(i)
                device_name = str(info["name"])
                max_channels = int(info["maxInputChannels"])
                if microphone_name.lower() in device_name.lower() and max_channels > 0:
                    logging.info(
                        f"[Whisper] Found microphone '{info['name']}' at index {i}"
                    )
                    return i
            logging.warning(
                f"[Whisper] Microphone '{microphone_name}' not found, using default"
            )

        return None

    def _audio_capture_loop(self) -> None:
        """
        Continuously capture audio from the microphone and process chunks.

        Runs in a separate thread. Reads audio chunks, performs silence
        detection, and triggers transcription when speech followed by
        silence is detected.
        """
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._chunk,
            )
            logging.info("[Whisper] Audio capture started")

            while self.running:
                try:
                    audio_data = self._stream.read(
                        self._chunk, exception_on_overflow=False
                    )
                except OSError as e:
                    logging.warning(f"[Whisper] Audio read error: {e}")
                    continue

                self._on_audio_data(audio_data)

        except Exception as e:
            logging.error(f"[Whisper] Audio capture error: {e}")
        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None

    def _on_audio_data(self, audio_data: bytes) -> None:
        """
        Process incoming raw audio data.

        Accumulates audio chunks and triggers transcription when silence
        is detected after speech.

        Parameters
        ----------
        audio_data : bytes
            Raw PCM16 audio data from the microphone.
        """
        # Calculate RMS for silence detection
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

        with self._buffer_lock:
            self._audio_buffer.append(audio_data)

            if rms > self._silence_threshold:
                self._silent_chunks = 0
                self._has_speech = True
            else:
                self._silent_chunks += 1

            # Calculate how many chunks equal the silence duration
            chunks_per_second = self._rate / self._chunk
            silence_chunk_limit = int(self._silence_duration * chunks_per_second)

            if self._has_speech and self._silent_chunks >= silence_chunk_limit:
                audio_to_transcribe = b"".join(self._audio_buffer)
                self._audio_buffer.clear()
                self._silent_chunks = 0
                self._has_speech = False

                self._transcription_queue.put(audio_to_transcribe)

    def _transcription_worker(self) -> None:
        """
        Dedicated worker loop that processes transcription requests sequentially.

        Runs in a single thread, pulling audio data from the queue and
        transcribing one at a time. This serializes model access and prevents
        unbounded thread spawning.
        """
        while self.running:
            try:
                audio_data = self._transcription_queue.get(timeout=0.5)
                if audio_data is None:
                    break
                self._transcribe(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"[Whisper] Transcription worker error: {e}")

    def _transcribe(self, audio_data: bytes) -> None:
        """
        Transcribe audio data using the Whisper model.

        Parameters
        ----------
        audio_data : bytes
            Raw PCM16 audio bytes to transcribe.
        """
        try:
            # Convert bytes to float32 numpy array normalized to [-1, 1]
            audio_array = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Skip very short audio
            if len(audio_array) < self._rate * 0.3:
                return

            segments, info = self._model.transcribe(
                audio_array,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts).strip()

            if full_text and self._message_callback:
                message = json.dumps({"asr_reply": full_text})
                self._message_callback(message)
                detected_lang = info.language if info else "unknown"
                logging.info(f"[Whisper] Transcribed ({detected_lang}): {full_text}")

        except Exception as e:
            logging.error(f"[Whisper] Transcription error: {e}")

    def register_message_callback(self, message_callback: Optional[Callable]) -> None:
        """
        Register a callback for processing ASR results.

        Parameters
        ----------
        message_callback : Optional[Callable]
            The callback function to process ASR results.
        """
        if message_callback is not None:
            self._message_callback = message_callback

    def start(self) -> None:
        """
        Start the Whisper ASR provider.

        Initializes the audio capture thread for continuous transcription.
        """
        if self.running:
            logging.warning("Whisper ASR provider is already running")
            return

        self.running = True
        self._audio_thread = threading.Thread(
            target=self._audio_capture_loop, daemon=True
        )
        self._audio_thread.start()

        self._transcription_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._transcription_thread.start()

        logging.info("Whisper ASR provider started")

    def stop(self) -> None:
        """
        Stop the Whisper ASR provider.

        Stops audio capture, transcription worker, and releases PyAudio resources.
        """
        self.running = False

        self._transcription_queue.put(None)

        if self._audio_thread:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None

        if self._transcription_thread:
            self._transcription_thread.join(timeout=2.0)
            self._transcription_thread = None

        with self._buffer_lock:
            self._audio_buffer.clear()

        if self._pa:
            self._pa.terminate()

        logging.info("Whisper ASR provider stopped")
