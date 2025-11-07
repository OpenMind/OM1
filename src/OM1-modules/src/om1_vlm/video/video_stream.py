import asyncio
import base64
import inspect
import json
import logging
import platform
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2

from .video_utils import enumerate_video_devices

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VideoStream:
    """
    Manages video capture and streaming from a camera device.

    Provides functionality to capture video frames from a camera device,
    process them, and stream them through a callback function. Supports
    both macOS and Linux camera devices.

    Parameters
    ----------
    frame_callback : Optional[Callable[[str], None]], optional
    frame_callbacks : Optional[List[Callable[[str], None]]], optional
        List of callback functions to be called with base64 encoded frame data,
        by default None
    fps : Optional[int], optional
        Frames per second to capture.
        By default 30
    resolution : Optional[Tuple[int, int]], optional
        Resolution of the captured video frames.
        By default (640, 480)
    jpeg_quality : int, optional
        JPEG quality for encoding frames, by default 70
    device_index : int, optional
        Index of the camera device to use, by default 0
    """

    def __init__(
        self,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
        device_index: int = 0,
    ):
        self._video_thread: Optional[threading.Thread] = None

        # Callbacks for video frame data
        self.frame_callbacks = frame_callbacks or []
        self.register_frame_callback(frame_callback)

        # Video capture device
        self.device_index = device_index
        self._cap = None

        self.running: bool = True

        self.fps = fps
        self.frame_delay = 1.0 / fps  # Calculate delay between frames
        self.resolution = resolution
        self.encode_quality = [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
        ]

        # Create a dedicated event loop for async tasks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

    def _start_loop(self):
        """Set and run the event loop forever in a dedicated thread."""
        asyncio.set_event_loop(self.loop)
        logger.debug("Starting background event loop for video streaming.")
        self.loop.run_forever()

    def on_video(self):
        """
        Main video capture and processing loop.

        Captures frames from the camera, encodes them to base64,
        and sends them through the callback if registered.

        Raises
        ------
        Exception
            If video streaming encounters an error
        """

        devices = enumerate_video_devices()
        if platform.system() == "Darwin":
            camindex = 0 if devices else 0
        else:
            camindex = "/dev/video" + str(devices[0][0]) if devices else "/dev/video0"

        if self.device_index != 0:
            if platform.system() == "Darwin":
                camindex = self.device_index
            else:
                camindex = f"/dev/video{self.device_index}"

        logger.info(f"Using camera: {camindex}")

        self._cap = cv2.VideoCapture(camindex)
        if not self._cap.isOpened():
            logger.error(f"Error opening video stream from {camindex}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
            logger.warning(
                f"Camera doesn't support resolution {self.resolution}. Using {(actual_width, actual_height)} instead."
            )
            self.resolution = (actual_width, actual_height)

        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        frame_time = 1.0 / self.fps
        last_frame_time = time.perf_counter()

        try:
            while self.running:
                current_time = time.perf_counter()
                elapsed = current_time - last_frame_time

                ret, frame = self._cap.read()
                if not ret:
                    logger.error("Error reading frame from video stream")
                    time.sleep(0.1)
                    continue

                if elapsed <= 1.5 * frame_time and self.frame_callbacks:
                    _, buffer = cv2.imencode(".jpg", frame, self.encode_quality)
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")

                    frame_data = json.dumps(
                        {"timestamp": time.time(), "frame": frame_base64}
                    )

                    for frame_callback in self.frame_callbacks:
                        if inspect.iscoroutinefunction(frame_callback):
                            asyncio.run_coroutine_threadsafe(
                                frame_callback(frame_data), self.loop
                            )
                        else:
                            frame_callback(frame_data)

                elapsed_time = time.perf_counter() - last_frame_time
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                last_frame_time = time.perf_counter()

        except Exception as e:
            logger.error(f"Error streaming video: {e}")
        finally:
            if self._cap:
                self._cap.release()
                logger.info("Released video capture device")

    def _start_video_thread(self):
        """
        Initialize and start the video processing thread.

        Creates a new daemon thread for video processing if one isn't
        already running.
        """
        if self._video_thread is None or not self._video_thread.is_alive():
            self._video_thread = threading.Thread(target=self.on_video, daemon=True)
            self._video_thread.start()
            logger.info("Started video processing thread")

    def register_frame_callback(self, frame_callback: Callable[[str], None]):
        """
        Register a callback function for processed frames.

        Parameters
        ----------
        frame_callback : Callable[[str], None]
            Function to be called with base64 encoded frame data
        """
        if frame_callback is None:
            logger.warning("Frame callback is None, not registering")
            return

        if frame_callback not in self.frame_callbacks:
            self.frame_callbacks.append(frame_callback)
            logger.info("Registered new frame callback")
            return

        logger.warning("Frame callback already registered")
        return

    def start(self):
        """
        Start video capture and processing.

        Initializes the video processing thread and begins
        capturing frames.
        """
        self._start_video_thread()

    def stop(self):
        """
        Stop video capture and clean up resources.

        Stops the video processing loop and waits for the
        processing thread to finish.
        """
        self.running = False

        if self._cap:
            self._cap.release()

        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)

        logger.info("Stopped video processing thread")
