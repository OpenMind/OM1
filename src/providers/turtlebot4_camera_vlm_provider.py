import base64
import logging
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import zenoh
from om1_utils import ws
from om1_vlm import VideoStream

from zenoh_msgs import open_zenoh_session

from .singleton import singleton


class TurtleBot4CameraVideoStream(VideoStream):
    """
    Video Stream class for TurtleBot4 camera.

    This class extends the VideoStream class to handle TurtleBot camera-specific
    video streaming and processing.

    NOTE - this uses the default preview image size of 250x250.
    If you want to stream high resolution data, you will need to change
    configurations inside the TurtleBot4.

    Please see here for instructions:
    https://github.com/turtlebot/turtlebot4/issues/335#issuecomment-1900339747
    """

    def __init__(
        self,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
        URID: str = "default",
        debug: bool = False,
    ):
        """
        Initialize the TurtleBot4 Camera Video Stream.

        Parameters
        ----------
        frame_callback : callable, optional
            A callback function to process video frames.
        frame_callbacks : list of callables, optional
            A list of callback functions to process video frames.
        fps : int, optional
            Frames per second for the video stream. Default is 30.
        resolution : tuple of int, optional
            The resolution for the video stream. Default is (640, 480).
        jpeg_quality : int, optional
            The JPEG quality for the video stream. Default is 70.
        URID : str, optional
            The URID for the Zenoh session. Default is "default".
        debug : bool, optional
            Enable debug mode for writing images to local files. Default is False.
        """
        super().__init__(
            frame_callback=frame_callback,
            frame_callbacks=frame_callbacks,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )
        self.session = None

        try:
            self.session = open_zenoh_session()
            topic = f"{URID}/pi/oakd/rgb/preview/image_raw"
            logging.info(
                f"TurtleBot4 Camera listener starting with URID: {URID} and topic: {topic}"
            )
            self.camera = self.session.declare_subscriber(topic, self.camera_listener)
            logging.info("Zenoh TurtleBot4 Camera subscriber created")
        except Exception as e:
            logging.error(f"Error opening Zenoh TurtleBot4 Camera client: {e}")

        self.lock = threading.Lock()
        self.image = None
        self.debug = debug

    def camera_listener(self, sample: zenoh.Sample):
        """
        Zenoh listener for incoming camera samples.

        Parameters
        ----------
        sample : zenoh.Sample
            The incoming sample from the Zenoh session.
        """
        bytesI = sample.payload.to_bytes()
        logging.debug(f"TurtleBot4 listener received {len(sample.payload)}")
        if bytesI and len(bytesI) == 187576:
            X = np.frombuffer(bytesI, dtype=np.uint8)
            # The first 76 numbers are some sort of metadata header?
            Xc = X[76:187576]
            rgb = np.reshape(Xc, (250, 250, 3))
            with self.lock:
                self.image = rgb
            if self.debug:
                cv2.imwrite("turtlebot.jpg", rgb)

    def on_video(self):
        """
        Main video capture and processing loop for TurtleBot cameras.

        Captures frames from the camera, encodes them to base64,
        and sends them through the callback if registered.
        """
        logging.info("TurtleBot Camera Video Stream")

        frame_time = 1.0 / (self.fps or 30)
        last_frame_time = time.perf_counter()

        while self.running:
            try:
                with self.lock:
                    image = self.image

                if image is not None:
                    resized_image = cv2.resize(
                        image, self.resolution, interpolation=cv2.INTER_AREA
                    )
                    _, buffer = cv2.imencode(".jpg", resized_image, self.encode_quality)
                    frame_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

                    if self.frame_callbacks:
                        for frame_callback in self.frame_callbacks:
                            frame_callback(frame_data)

                    with self.lock:
                        self.image = None

                elapsed_time = time.perf_counter() - last_frame_time
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                last_frame_time = time.perf_counter()

            except Exception as e:
                logging.error(f"Error in video processing loop: {e}")
                continue

        logging.info("Stopping Camera Video Stream")


@singleton
class TurtleBot4CameraVLMProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage camera input streaming
    and websocket communication for vlm services. It runs in a separate thread
    to handle continuous vlm processing.
    """

    def __init__(
        self,
        ws_url: str,
        fps: int = 15,
        resolution: Optional[Tuple[int, int]] = (480, 480),
        jpeg_quality: int = 70,
        URID: str = "default",
        stream_url: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        fps : int, optional
            The fps for the VLM service connection. Default is 15.
        resolution : tuple of int, optional
            The resolution for the video stream. Default is (480, 480).
        jpeg_quality : int, optional
            The JPEG quality for the video stream. Default is 70.
        URID : str, optional
            The URID for the Zenoh session. Default is "default".
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        debug : bool, optional
            Enable debug mode for writing images to local files. Default is False.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: VideoStream = TurtleBot4CameraVideoStream(
            self.ws_client.send_message,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            URID=URID,
            debug=debug,
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
            logging.warning("TurtleBot4 Camera VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("TurtleBot4 Camera VLM provider started")

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
