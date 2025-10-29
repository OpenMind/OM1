"""
This script is used to test the remote audio input functionality of a system.
"""

import base64
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import pyaudio
from om1_utils import ws

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OM_API_KEY = os.environ.get("OM_API_KEY", None)
if not OM_API_KEY:
    logging.error("OM_API_KEY environment variable not set.")
    sys.exit(1)


@dataclass
class AudioInput:
    """
    Data class to represent audio input parameters for remote audio input.
    """

    audio: str
    rate: int
    language_code: str = "en-US"

    def to_dict(self) -> dict:
        """
        Convert the AudioInput object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the AudioInput object.
        """
        return {
            "audio": self.audio,
            "rate": self.rate,
            "language_code": self.language_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioInput":
        """
        Populate the AudioInput object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing audio input parameters.
        """
        return cls(
            audio=data.get("audio", ""),
            rate=data.get("rate", 16000),
            language_code=data.get("language_code", "en-US"),
        )


class RemoteAudioInput:
    def __init__(
        self,
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        device: Optional[int] = None,
        device_name: str = None,
    ):
        self.ws_client: ws.Client = ws.Client(
            url=f"wss://api.openmind.org/api/core/teleops/audio?api_key={OM_API_KEY}"
        )
        self.ws_client.start()

        self._rate = rate
        self._chunk = chunk
        self._device = device
        self._device_name = device_name

        self._audio_interface: pyaudio.PyAudio = pyaudio.PyAudio()
        self._audio_stream: Optional[pyaudio.Stream] = None

        self._audio_thread: Optional[threading.Thread] = None

        self.running: bool = True

        try:
            input_device = None
            device_count = self._audio_interface.get_device_count()
            logging.info(f"Found {device_count} audio devices")

            if self._device is not None:
                input_device = self._audio_interface.get_device_info_by_index(
                    self._device
                )
                logging.info(
                    f"Selected input device: {input_device['name']} ({self._device})"
                )
                if input_device["maxInputChannels"] == 0:
                    logging.warning(
                        f"Selected input device does not advertise input channels: {input_device['name']} ({self._device})"
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
                logging.info(
                    f"Default input device: {input_device['name']} ({self._device})"
                )

            if input_device is None:
                raise ValueError("No input device found")

            logging.info(
                f"Selected input device: {input_device['name']} ({self._device})"
            )

            if rate is None:
                self._rate = int(input_device.get("defaultSampleRate", 16000))
                logging.info(f"Using device default sample rate: {self._rate} Hz")
            else:
                self._rate = rate
                logging.info(f"Using specified sample rate: {self._rate} Hz")

            if chunk is None:
                self._chunk = int(self._rate * 0.2)  # ~200ms of audio
                logging.info(
                    f"Using calculated chunk size: {self._chunk} frames (~200ms)"
                )
            else:
                self._chunk = chunk
                chunk_duration_ms = (self._chunk / self._rate) * 1000
                logging.info(
                    f"Using specified chunk size: {self._chunk} frames (~{chunk_duration_ms:.1f}ms)"
                )

        except Exception as e:
            logging.error(f"Error initializing audio input: {e}")
            self._audio_interface.terminate()
            raise

    def start(self) -> "RemoteAudioInput":
        """
        Initializes and starts the audio capture stream.

        This method sets up the PyAudio interface, configures the input device,
        and starts the audio processing thread.

        Returns
        -------
        RemoteAudioInput
            The current instance for method chaining

        Raises
        ------
        Exception
            If there are errors initializing the audio interface or opening the stream
        """
        if not self.running:
            return self

        try:
            self._audio_stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                input_device_index=self._device,
                channels=1,
                rate=self._rate,
                input=True,
                frames_per_buffer=self._chunk,
                stream_callback=self._fill_buffer,
            )

            logging.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logging.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def _fill_buffer(self, in_data: bytes, frame_count: int, time_info, status_flags):
        """
        Callback function to handle incoming audio data.

        Parameters
        ----------
        in_data : bytes
            Raw audio data from the input stream
        frame_count : int
            Number of frames in the incoming data
        time_info : dict
            Timing information from the audio stream
        status_flags : int
            Status flags indicating stream status

        Returns
        -------
        tuple
            A tuple containing the processed audio data and a flag indicating whether to continue the stream
        """
        if self.ws_client.is_connected():
            encode_audio = base64.b64encode(in_data).decode("utf-8")
            audio_input = AudioInput(
                audio=encode_audio, rate=self._rate, language_code="en-US"
            )
            self.ws_client.send_message(json.dumps(audio_input.to_dict()))
        return None, pyaudio.paContinue

    def _start_audio_thread(self):
        """
        Starts the audio processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread = threading.Thread(target=self.on_audio, daemon=True)
            self._audio_thread.start()
            logging.info("Started audio processing thread")


if __name__ == "__main__":
    try:
        remote_audio_input = RemoteAudioInput(rate=48000, chunk=12144)
        remote_audio_input.start()
        logging.info("Remote audio input started successfully.")

        while remote_audio_input.running:
            try:
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                remote_audio_input.running = False

    except Exception as e:
        logging.error(f"Failed to start remote audio input: {e}")
