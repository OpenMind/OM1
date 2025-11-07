import argparse
import base64
import json
import logging
import shutil
import subprocess
import threading
import time
from queue import Queue
from typing import Callable, Dict, Optional

import requests

from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioOutputStream:
    """
    A class for managing audio output and text-to-speech (TTS) conversion.

    Parameters
    ----------
    url : str
        The URL endpoint for the text-to-speech service
    rate : int, optional
        The sampling rate in Hz for audio output (default: 8000)
    tts_state_callback : Optional[Callable], optional
        A callback function to receive TTS state changes (active/inactive)
        (default: None)
    headers : Optional[Dict[str, str]], optional
        Additional headers to include in the HTTP request (default: None)
    """

    def __init__(
        self,
        url: str,
        rate: int = 8000,
        tts_state_callback: Optional[Callable] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self._url = url
        self._rate = rate

        # Process headers
        self._headers = headers or {}
        if "Content-Type" not in self._headers:
            self._headers["Content-Type"] = "application/json"

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

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

        # Pending requests queue
        self._pending_requests: Queue[Optional[str]] = Queue()

        # Slience audio for Bluetooth optimization
        self._silence_audio = self._create_silence_audio(100)
        self._silence_prefix = self._create_silence_audio(500)

        # Running state and last audio time
        self.running: bool = True
        self._last_audio_time = time.time()

    def zenoh_audio_message(self, data: str):
        """
        Callback function for Zenoh audio status messages.

        Parameters
        ----------
        data : str
            The audio data in base64 encoded string format.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())
        if (
            self.audio_status
            and self.audio_status.sentence_to_speak.data
            and self.audio_status.status_speaker
            == AudioStatus.STATUS_SPEAKER.ACTIVE.value
        ):
            pending_message = json.loads(self.audio_status.sentence_to_speak.data)
            self.add_request(pending_message)

    def set_tts_state_callback(self, callback: Callable):
        """
        Set a callback function for TTS state changes.

        Parameters
        ----------
        callback : Callable
            Function to be called when TTS state changes (active/inactive)
        """
        self._tts_state_callback = callback

    def add_request(self, audio_request: Dict[str, str]):
        """
        Add request to the TTS processing queue.

        Parameters
        ----------
        audio_request : Dict[str, str]
            Request to be processed by the TTS service
        """
        self._pending_requests.put(audio_request)

    def _process_audio(self):
        """
        Process the TTS queue and play audio output.

        Makes HTTP requests to the TTS service, converts responses to audio,
        and plays them through the audio device.
        """
        while self.running:
            try:
                tts_request = self._pending_requests.get()
                response = requests.post(
                    self._url,
                    data=json.dumps(tts_request),
                    headers=self._headers,
                    timeout=(5, 15),
                )
                logger.info(f"Received TTS response: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        audio_data = result["response"]
                        self._write_audio(audio_data)
                        logger.debug(f"Processed TTS request: {tts_request}")
                else:
                    logger.error(
                        f"TTS request failed with status code {response.status_code}: {response.text}. Request details: {tts_request}"
                    )
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def _create_silence_audio(self, duration_ms: int = 500) -> bytes:
        """
        Create silent audio data.

        Parameters
        ----------
        duration_ms : int
            Duration of silence in milliseconds

        Returns
        -------
        bytes
            Base64 encoded silent audio data
        """
        samples = int(self._rate * duration_ms / 1000)
        silence_bytes = b"\x00" * (samples * 2)
        return base64.b64encode(silence_bytes)

    def _keepalive_worker(self):
        """
        Background thread to play keepalive sounds every 60 seconds.
        """
        while self.running:
            current_time = time.time()
            if current_time - self._last_audio_time >= 60:
                self._write_audio_bytes(self._silence_audio, is_keepalive=True)
                self._last_audio_time = current_time
            time.sleep(10)

    def _write_audio(self, audio_data: bytes):
        """
        Write audio data to the output stream with Bluetooth optimization.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        """
        self._last_audio_time = time.time()

        audio_bytes = self._silence_prefix + base64.b64decode(audio_data)

        self._write_audio_bytes(base64.b64encode(audio_bytes))

    def _write_audio_bytes(self, audio_data: bytes, is_keepalive: bool = False):
        """
        Write audio data to the output stream using ffplay.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        is_keepalive : bool
            Whether this is a keepalive sound (suppresses callbacks)
        """
        audio_bytes = base64.b64decode(audio_data)

        if not is_installed("ffplay"):
            message = (
                "ffplay from ffmpeg not found, necessary to play audio. "
                "On mac you can install it with 'brew install ffmpeg'. "
                "On linux and windows you can install it from https://ffmpeg.org/"
            )
            logger.error(message)
            return

        if not is_keepalive:
            self._tts_callback(True)

            state = AudioStatus(
                header=prepare_header(),
                status_mic=(
                    self.audio_status.status_mic
                    if self.audio_status
                    else AudioStatus.STATUS_MIC.UNKNOWN.value
                ),
                status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
                sentence_to_speak=String(""),
            )

            if self.pub:
                self.pub.put(state.serialize())

        args = [
            "ffplay",
            "-autoexit",
            "-",
            "-nodisp",
            # It is not good at supporting audio_device_index from pyaudio
            # Reading the list from ffplay doesn't work either
            # "-audio_device_index",
            # str(self._device),
        ]
        proc = subprocess.Popen(
            args=args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.communicate(input=audio_bytes)
        exit_code = proc.poll()

        if exit_code != 0 and not is_keepalive:
            logger.error(f"Error playing audio: {exit_code}")

        if not is_keepalive:
            self._tts_callback(False)

            state = AudioStatus(
                header=prepare_header(),
                status_mic=(
                    self.audio_status.status_mic
                    if self.audio_status
                    else AudioStatus.STATUS_MIC.UNKNOWN.value
                ),
                status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
                sentence_to_speak=String(""),
            )

            if self.pub:
                self.pub.put(state.serialize())

    def _tts_callback(self, is_active: bool):
        """
        Invoke the TTS state callback if set.

        Parameters
        ----------
        is_active : bool
            Whether TTS is currently active
        """
        if self._tts_state_callback:
            self._tts_state_callback(is_active)

    def start(self):
        """
        Start the audio processing thread.

        Initializes a daemon thread for processing the TTS queue.
        """
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()

        keepalive_thread = threading.Thread(target=self._keepalive_worker)
        keepalive_thread.daemon = True
        keepalive_thread.start()

    def run_interactive(self):
        """
        Run an interactive console for text-to-speech conversion.

        Allows users to input text for TTS conversion until 'quit' is entered
        or KeyboardInterrupt is received.
        """
        logger.info(
            "Running interactive audio output stream. Please enter text for TTS conversion."
        )
        try:
            while self.running:
                user_input = input()
                if user_input.lower() == "quit":
                    break
                self.add_request({"text": user_input})
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()

    def stop(self):
        """
        Stop the audio output stream and cleanup resources.
        """
        self.running = False

        if self.session:
            self.session.close()


def is_installed(lib_name: str) -> bool:
    return shutil.which(lib_name) is not None


def main():
    """
    Main function for running the audio output stream.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-url", type=str, required=True, help="URL for the TTS service"
    )
    args = parser.parse_args()

    audio_output = AudioOutputStream(args.tts_url, device=args.device, rate=args.rate)
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()


if __name__ == "__main__":
    main()
