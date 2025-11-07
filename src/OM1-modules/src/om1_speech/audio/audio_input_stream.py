# Description: Audio stream class for capturing audio from a microphone
# A partial of code comes from https://github.com/nvidia-riva/python-clients/blob/main/riva/client/audio_io.py

import base64
import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import pyaudio

from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioInputStream:
    """
    A class for capturing and managing real-time audio input from a microphone.

    This class provides functionality to capture audio data from a specified or default
    microphone device, process it in chunks, and make it available through a generator
    or callback mechanism. It supports Text-to-Speech (TTS) integration by temporarily
    suspending audio capture during TTS playback.

    Parameters
    ----------
    rate : int, optional
        The sampling rate in Hz for audio capture. If None, uses the default rate of the selected device.
        (default: None)
    chunk : int, optional
        The size of each audio chunk in frames. If None, automatically calculates an optimal chunk size
        based on the sample rate (approximately 100ms of audio).
        (default: None)
    device : Optional[Union[str, int, float, Any]], optional
        The input device identifier. Can be device index or name. If None,
        uses system default input device (default: None)
    device_name: str, optional
        The input device name. If None, uses the first available input device.
        (default: None)
    audio_data_callback : Optional[Callable], optional
        A callback function that receives audio data chunks (default: None)
    language_code: str, optional
        The language for the ASR to listen. (default: en-US)
    remote_input : bool, optional
        If True, indicates that the audio input is from a remote source.
    """

    def __init__(
        self,
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        device: Optional[Union[str, int, float, Any]] = None,
        device_name: str = None,
        audio_data_callback: Optional[Callable] = None,
        audio_data_callbacks: Optional[List[Callable]] = None,
        language_code: Optional[str] = None,
        remote_input: bool = False,
    ):
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self._device_name = device_name

        # Determine language code
        if language_code is None:
            self._language_code = "en-US"
            logger.info(f"Using default language code: {self._language_code}")
        else:
            self._language_code = language_code
            logger.info(f"Using specified language code: {self._language_code}")

        # Callback for audio data
        self._audio_data_callbacks = audio_data_callbacks or []
        self.register_audio_data_callback(audio_data_callback)

        # Flag to indicate if TTS is active
        self._is_tts_active: bool = False

        # Thread-safe buffer for audio data
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()

        # audio interface and stream
        self._audio_interface: pyaudio.PyAudio = pyaudio.PyAudio()
        self._audio_stream: Optional[pyaudio.Stream] = None

        # Audio processing thread
        self._audio_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        # Zenoh
        self.topic = "robot/status/audio"
        self.session = None
        self.pub = None
        self.audio_status = None

        try:
            self.session = open_zenoh_session()
            self.pub = self.session.declare_publisher(self.topic)
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)
        except Exception as e:
            logger.error(f"Failed to declare Zenoh subscriber: {e}")
            self.session = None

        self.running: bool = True

        self.remote_input = remote_input
        if self.remote_input:
            logger.info("Remote input is enabled, skipping audio input initialization")
            return

        if self._device is not None and self._device_name is not None:
            raise ValueError("Only one of device or device_name can be specified")

        try:
            input_device = None
            device_count = self._audio_interface.get_device_count()
            logger.info(f"Found {device_count} audio devices")

            if self._device is not None:
                input_device = self._audio_interface.get_device_info_by_index(
                    self._device
                )
                logger.info(
                    f"Selected input device: {input_device['name']} ({self._device})"
                )
                if input_device["maxInputChannels"] == 0:
                    logger.warning(
                        f"Selected input device does not advertize input channels: {input_device['name']} ({self._device})"
                    )
            elif self._device_name is not None:
                available_devices = []
                for i in range(device_count):
                    device_info = self._audio_interface.get_device_info_by_index(i)
                    device_name = device_info["name"]
                    available_devices.append({"name": device_name, "index": i})
                    if self._device_name.lower() in device_name.lower():
                        input_device = device_info
                        self._device = i
                        break
                if input_device is None:
                    raise ValueError(
                        f"Input device '{self._device_name}' not found. Available devices: {available_devices}"
                    )
            else:
                input_device = self._audio_interface.get_default_input_device_info()
                self._device = input_device["index"]
                logger.info(
                    f"Default input device: {input_device['name']} ({self._device})"
                )

            if input_device is None:
                raise ValueError("No input device found")

            logger.info(
                f"Selected input device: {input_device['name']} ({self._device})"
            )

            if rate is None:
                self._rate = int(input_device.get("defaultSampleRate", 16000))
                logger.info(f"Using device default sample rate: {self._rate} Hz")
            else:
                self._rate = rate
                logger.info(f"Using specified sample rate: {self._rate} Hz")

            if chunk is None:
                self._chunk = int(self._rate * 0.2)  # ~200ms of audio
                logger.info(
                    f"Using calculated chunk size: {self._chunk} frames (~200ms)"
                )
            else:
                self._chunk = chunk
                chunk_duration_ms = (self._chunk / self._rate) * 1000
                logger.info(
                    f"Using specified chunk size: {self._chunk} frames (~{chunk_duration_ms:.1f}ms)"
                )

        except Exception as e:
            logger.error(f"Error initializing audio input: {e}")
            self._audio_interface.terminate()
            raise

    def zenoh_audio_message(self, data: str):
        """
        Callback function for Zenoh audio status messages.

        Parameters
        ----------
        data : str
            The audio data in base64 encoded string format.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

        if self.audio_status.status_speaker == AudioStatus.STATUS_SPEAKER.ACTIVE.value:
            with self._lock:
                if not self._is_tts_active:
                    state = AudioStatus(
                        header=prepare_header(),
                        status_mic=self.audio_status.status_mic,
                        status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
                        sentence_to_speak=String(""),
                    )

                    if self.pub:
                        self.pub.put(state.serialize())

                    self._is_tts_active = True

        if self.audio_status.status_speaker == AudioStatus.STATUS_SPEAKER.READY.value:
            with self._lock:
                if self._is_tts_active:
                    state = AudioStatus(
                        header=prepare_header(),
                        status_mic=self.audio_status.status_mic,
                        status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
                        sentence_to_speak=String(""),
                    )

                    if self.pub:
                        self.pub.put(state.serialize())

                    self._is_tts_active = False

    def on_tts_state_change(self, is_active: bool):
        """
        Updates the TTS active state to control audio capture behavior.

        When TTS is active, audio capture is temporarily suspended to prevent
        capturing the TTS output.

        Parameters
        ----------
        is_active : bool
            True if TTS is currently playing, False otherwise
        """
        with self._lock:
            self._is_tts_active = is_active
            logger.info(f"TTS active state changed to: {is_active}")

    def register_audio_data_callback(self, audio_callback: Callable):
        """
        Registers a callback function for audio data processing.

        Parameters
        ----------
        callback : Callable
            Function to be called with audio data chunks
        """
        if audio_callback is None:
            logger.warning("Audio data callback is None, not registering")
            return

        if audio_callback not in self._audio_data_callbacks:
            self._audio_data_callbacks.append(audio_callback)
            logger.info("Registered new audio data callback")
            return

        logger.warning("Audio data callback already registered")
        return

    def start(self) -> "AudioInputStream":
        """
        Initializes and starts the audio capture stream.

        This method sets up the PyAudio interface, configures the input device,
        and starts the audio processing thread.

        Returns
        -------
        AudioInputStream
            The current instance for method chaining

        Raises
        ------
        Exception
            If there are errors initializing the audio interface or opening the stream
        """
        if not self.running:
            return self

        try:
            if not self.remote_input:
                logger.info("Remote input is disabled, initializing audio input stream")
                self._audio_stream = self._audio_interface.open(
                    format=pyaudio.paInt16,
                    input_device_index=self._device,
                    channels=1,
                    rate=self._rate,
                    input=True,
                    frames_per_buffer=self._chunk,
                    stream_callback=self._fill_buffer,
                )

            # Start the audio processing thread
            self._start_audio_thread()

            logger.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def _start_audio_thread(self):
        """
        Starts the audio processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread = threading.Thread(target=self.on_audio, daemon=True)
            self._audio_thread.start()
            logger.info("Started audio processing thread")

    def _fill_buffer(
        self, in_data: bytes, frame_count: int, time_info: dict, status_flags: int
    ) -> Tuple[None, int]:
        """
        Callback function for the PyAudio stream to fill the audio buffer.

        This method is called by PyAudio when new audio data is available.
        It adds the data to the buffer queue if TTS is not active.

        Parameters
        ----------
        in_data : bytes
            The captured audio data
        frame_count : int
            Number of frames in the audio data
        time_info : dict
            Timing information from PyAudio
        status_flags : int
            Status flags from PyAudio

        Returns
        -------
        Tuple[None, int]
            A tuple containing None and pyaudio.paContinue
        """
        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)
        return None, pyaudio.paContinue

    def fill_buffer_remote(self, data: str):
        """
        Callback function for remote audio data to fill the audio buffer.

        This method is called when remote audio data is received. It adds the
        data to the buffer queue if TTS is not active.

        Parameters
        ----------
        data : str
        """
        try:
            audio_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON data: {e}")
            return

        if "audio" not in audio_data:
            logger.error("Received remote audio data without 'audio' key")
            return

        in_data = base64.b64decode(audio_data["audio"])
        rate = audio_data.get("rate", self._rate)
        language_code = audio_data.get("language_code", self._language_code)

        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)
                self._rate = rate
                self._language_code = language_code

    def generator(self) -> Generator[Dict[str, Union[bytes, int]], None, None]:
        """
        Generates a stream of audio data chunks.

        This generator yields audio data chunks, combining multiple chunks when
        available to reduce processing overhead. It skips yielding data when
        TTS is active.

        Yields
        ------
        bytes
            Combined audio data chunks
        """
        while self.running:
            chunk = self._buff.get()
            if chunk is None:
                return

            with self._lock:
                if self._is_tts_active:
                    continue

            # Collect additional chunks that are immediately available
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        assert self.running
                    if chunk:
                        data.append(chunk)
                except queue.Empty:
                    break

            response = {
                "audio": base64.b64encode(b"".join(data)).decode("utf-8"),
                "rate": self._rate,
                "language_code": self._language_code,
            }
            for audio_callback in self._audio_data_callbacks:
                audio_callback(json.dumps(response))

            yield response

    def on_audio(self):
        """Audio processing loop"""
        for _ in self.generator():
            if not self.running:
                break
        pass

    def stop(self):
        """
        Stops the audio stream and cleans up resources.

        This method stops the audio stream, terminates the PyAudio interface,
        and ensures the audio processing thread is properly shut down.
        """
        self.running = False

        if self.session:
            self.session.close()

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio_interface:
            self._audio_interface.terminate()

        # Clean up the audio processing thread
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

        self._buff.put(None)
        logger.info("Stopped audio stream")
