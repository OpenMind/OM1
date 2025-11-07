import asyncio
import base64
import inspect
import logging
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from google.protobuf import text_format

from ..gz.msgs import image_pb2

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class GazeboVideoStream:
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
        By default (480, 270)
    topic : Optional[str], optional
        Gazebo topic to obtain simulated camera feed.
        By default /camera
    """

    def __init__(
        self,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (480, 270),
        topic: Optional[str] = "/camera",
    ):
        self._video_thread: Optional[threading.Thread] = None

        # Callbacks for video frame data
        self.frame_callbacks = frame_callbacks or []
        self.register_frame_callback(frame_callback)

        # Video capture device
        self._cap = None
        self.capture_topic = topic

        self.running: bool = True

        self.fps = fps
        self.frame_delay = 1.0 / fps  # Calculate delay between frames
        self.resolution = resolution

        # Create a dedicated event loop for async tasks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

    def _start_loop(self):
        """Set and run the event loop forever in a dedicated thread."""
        asyncio.set_event_loop(self.loop)
        logger.debug("Starting background event loop for video streaming.")
        self.loop.run_forever()

    def _parse_text_message(self, text_data):
        """
        Parses a text-formatted protobuf message (with escaped binary data)
        into an Image message.
        """
        img_msg = image_pb2.Image()
        try:
            # The text_format parser automatically unescapes bytes fields.
            text_format.Parse(text_data, img_msg)
        except Exception as e:
            logging.error("Text parsing failed:", e)
            return None
        return img_msg

    def _process_image(self, img_msg):
        """
        Converts the protobuf Image message into a NumPy array, taking into account
        the row stride (step) and converting color channels for OpenCV.
        """
        width = img_msg.width
        height = img_msg.height
        step = img_msg.step  # Number of bytes per row (may include padding)
        encoding = img_msg.pixel_format_type
        raw_data = img_msg.data  # The text parser converts the escaped string to bytes

        # Determine number of channels based on pixel format.
        if encoding == image_pb2.PixelFormatType.RGB_INT8:
            channels = 3
        elif encoding == image_pb2.PixelFormatType.BGR_INT8:
            channels = 3
        elif encoding == image_pb2.PixelFormatType.RGBA_INT8:
            channels = 4
        elif encoding == image_pb2.PixelFormatType.GRAY8:
            channels = 1
        else:
            logging.error(f"Unsupported pixel format: {encoding}")
            return None

        expected_bytes_per_row = width * channels
        if step < expected_bytes_per_row:
            logging.error(
                f"Step value ({step}) is less than expected row bytes ({expected_bytes_per_row})."
            )
            return None

        try:
            if step == expected_bytes_per_row:
                image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (height, width, channels)
                )
            else:
                # Reshape to (height, step) then crop each row to the actual image data.
                rows = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, step))
                image_array = rows[:, :expected_bytes_per_row].reshape(
                    (height, width, channels)
                )
        except Exception as e:
            logging.error("Error reshaping image data:", e)
            return None

        # Convert color channels as needed. OpenCV expects BGR.
        if encoding == image_pb2.PixelFormatType.RGB_INT8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif encoding == image_pb2.PixelFormatType.RGBA_INT8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif encoding == image_pb2.PixelFormatType.GRAY8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        # BGR_INT8: no conversion needed

        return image_array

    def _get_message(self):
        """
        Runs gz topic to fetch one message from /camera in text format.
        """
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-t", self.capture_topic, "-n", "1"],
                capture_output=True,
                text=True,
                timeout=5,  # Avoid indefinite hangs
            )

            if result.returncode != 0:
                logging.error(f"Command failed with error: {result.stderr}")
                return None

            return result.stdout

        except subprocess.TimeoutExpired:
            logging.error("Subprocess timed out while fetching message from /camera.")
            return None

        except FileNotFoundError:
            logging.error(
                "The 'gz' command was not found. Ensure that Gazebo is installed and in PATH."
            )
            return None

        except Exception as e:
            logging.error(f"Unexpected error in _get_message: {e}")
            return None

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

        try:
            frame_time = 1.0 / self.fps
            last_frame_time = time.perf_counter()

            while self.running:
                text_data = self._get_message()

                if not text_data:
                    return
                # Parse the text-formatted protobuf message.
                img_msg = self._parse_text_message(text_data)
                if img_msg is None:
                    logging.debug("Unable to parse current image message")
                    return
                # Process the image from the parsed message.
                frame = self._process_image(img_msg)
                if frame is None:
                    logging.debug("Unable to process current image message")
                    return

                resized_frame = cv2.resize(frame, self.resolution)

                # Convert frame to base64
                _, buffer = cv2.imencode(
                    ".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                frame_data = base64.b64encode(buffer).decode("utf-8")

                if self.frame_callbacks:
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

        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)

        logger.info("Stopped video processing thread")
