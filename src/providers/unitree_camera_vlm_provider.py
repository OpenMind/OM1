import base64
import logging
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from om1_utils import ws
from om1_vlm import VideoStream

from .singleton import singleton

try:
    from unitree.unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError:
    logging.warning(
        "Unitree SDK not found. Please install the Unitree SDK to use this plugin."
    )


class UnitreeCameraVideoStream(VideoStream):
    """
    Video Stream class for Unitree Cameras.

    This class extends the VideoStream class to handle Unitree camera-specific
    video streaming and processing.
    """

    def __init__(
        self,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
    ):
        """
        Initialize the Unitree Camera Video Stream.

        Parameters
        ----------
        frame_callback : callable, optional
            A callback function to process video frames.
        frame_callbacks : list of callables, optional
            A list of callback functions to process video frames.
        fps : int, optional
            Frames per second for the video stream.
        resolution : tuple of int, optional
            The resolution for the video stream.
        jpeg_quality : int, optional
            The JPEG quality for the video stream.
        """
        super().__init__(
            frame_callback=frame_callback,
            frame_callbacks=frame_callbacks,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

        self.video_client = VideoClient()  # type: ignore
        self.video_client.Init()

    def on_video(self):
        """
        Main video capture and processing loop for Unitree cameras.

        Captures frames from the camera, encodes them to base64,
        and sends them through the callback if registered.
        """
        logging.info("Starting Unitree Camera Video Stream")

        frame_time = 1.0 / (self.fps or 30)
        last_frame_time = time.perf_counter()

        while self.running:
            try:
                code, data = self.video_client.GetImageSample()
                if code == 0 and data is not None:
                    # Convert to numpy image
                    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                    if image is not None:
                        # Calculate new dimensions maintaining aspect ratio
                        height, width = image.shape[:2]
                        ratio = width / height

                        target_resolution = self.resolution or (640, 480)

                        if width > height:
                            new_width = target_resolution[0]
                            new_height = int(target_resolution[0] / ratio)
                        else:
                            new_height = target_resolution[1]
                            new_width = int(target_resolution[1] * ratio)

                        # Resize image
                        resized_image = cv2.resize(
                            image, (new_width, new_height), interpolation=cv2.INTER_AREA
                        )
                        _, buffer = cv2.imencode(
                            ".jpg", resized_image, self.encode_quality
                        )
                        frame_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

                        if self.frame_callbacks:
                            for frame_callback in self.frame_callbacks:
                                frame_callback(frame_data)
                    else:
                        logging.warning("Failed to decode image")
                else:
                    logging.error(f"Failed to get image sample, code: {code}")

                elapsed_time = time.perf_counter() - last_frame_time
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                last_frame_time = time.perf_counter()

            except Exception as e:
                logging.error(f"Error in video processing loop: {e}")
                continue

        logging.info("Stopping Camera Video Stream")


@singleton
class UnitreeCameraVLMProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage camera input streaming and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        base_url: str,
        fps: int = 60,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
        stream_url: Optional[str] = None,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        base_url : str
            The base URL for the VLM service connection.
        fps : int, optional
            The frames per second for the VLM service connection. Defaults to 15.
        resolution : tuple of int, optional
            The resolution for the video stream. Defaults to (640, 480).
        jpeg_quality : int, optional
            The JPEG quality for the video stream. Defaults to 70.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=base_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: VideoStream = UnitreeCameraVideoStream(
            self.ws_client.send_message,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : Optional[callable]
            The callback function to process VLM results.
        """
        if message_callback is not None:
            self.ws_client.register_message_callback(message_callback)

    def start(self):
        """
        Start the VLM provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("Unitree Camera VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("Unitree Camera VLM provider started")

    def stop(self):
        """
        Stop the VLM provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False
        self.video_stream.stop()
        self.ws_client.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()
