# Description: Audio device input class for capturing audio from a microphone
# A partial of code comes from https://github.com/nvidia-riva/python-clients/blob/main/riva/client/audio_io.py

import base64
import logging
import queue
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pyaudio

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioDeviceInput:
    """
    A class for capturing real-time audio input from a microphone device.

    Parameters
    ----------
    rate : int, optional
        The sampling rate in Hz for audio capture (default: 16000)
    chunk : int, optional
        The size of each audio chunk in frames (default: 4048)
    device : Optional[Union[str, int, float, Any]], optional
        The input device identifier. Can be device index or name. If None,
        uses system default input device (default: None)
    callback : Optional[Callable], optional
        A callback function to receive audio data chunks (default: None)
    """

    def __init__(
        self,
        rate: int = 16000,
        chunk: int = 4048,
        device: Optional[Union[str, int, float, Any]] = None,
        callback: Optional[Callable] = None,
    ):
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self.callback = callback

        # Thread-safe buffer for audio data
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()
        self.running: bool = True
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._audio_stream: Optional[pyaudio.Stream] = None

    def setup_audio_devices(self) -> "AudioDeviceInput":
        """
        Initialize and set up the audio capture devices.

        Returns
        -------
        AudioDeviceInput
            The current instance for method chaining

        Raises
        ------
        Exception
            If there are errors initializing the audio interface or opening the stream
        """
        if not self.running:
            return self

        self._audio_interface = pyaudio.PyAudio()

        # Get default device if none specified
        if self._device is None:
            try:
                default_info = self._audio_interface.get_default_input_device_info()
                self._device = default_info["index"]
            except Exception as e:
                logger.error(f"Error getting default input device: {e}")
                self._device = None
        else:
            device_info = self._audio_interface.get_device_info_by_index(self._device)
            logger.info(
                f"Selected input device: {device_info['name']} ({self._device})"
            )

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

            logger.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def stop(self):
        """
        Stop the audio capture and clean up resources.

        Stops the audio stream, closes the audio interface, and signals
        termination through the buffer queue.
        """
        self.running = False

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio_interface:
            self._audio_interface.terminate()

        self._buff.put(None)
        logger.info("Stopped audio stream")

    def _fill_buffer(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Dict[str, Any],
        status_flags: int,
    ) -> Tuple[None, int]:
        """
        Callback function for the PyAudio stream to fill the audio buffer.

        Parameters
        ----------
        in_data : bytes
            The captured audio data
        frame_count : int
            Number of frames in the audio data
        time_info : Dict[str, Any]
            Timing information from PyAudio
        status_flags : int
            Status flags from PyAudio

        Returns
        -------
        Tuple[None, int]
            A tuple containing None and pyaudio.paContinue
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def get_audio_chunk(self) -> Optional[Dict[str, Union[bytes, int]]]:
        """
        Get the next chunk of audio data from the buffer.

        This method retrieves audio chunks from the buffer queue and combines
        multiple chunks if immediately available to reduce processing overhead.

        Returns
        -------
        Optional[bytes]
            Combined audio data chunks or None if the stream is terminated
        """
        while self.running:
            chunk = self._buff.get()
            if chunk is None:
                return

            # Collect additional chunks that are immediately available
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            return {
                "audio": base64.b64encode(b"".join(data)).decode("utf-8"),
                "rate": self._rate,
            }
