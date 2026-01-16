import logging
from typing import Callable, Optional, Tuple

from om1_utils import ws

from .singleton import singleton
from .ubtech_video_stream import UbtechCameraVideoStream


@singleton
class UbtechVLMProvider:
    """
    Singleton class to handle Ubtech video input and WebSocket communication for VLM.
    """

    def __init__(
        self,
        ws_url: str,
        robot_ip: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
        jpeg_quality: int = 70,
        stream_url: Optional[str] = None,
    ):
        """
        Initialize the Ubtech VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        robot_ip : str
            The IP address of the Ubtech robot.
        fps : int, optional
            The frames per second for the VLM service connection (default is 30).
        resolution : tuple of int, optional
            The resolution for the video stream (default is (640, 480)).
        jpeg_quality : int, optional
            The JPEG quality for the video stream (default is 70).
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        """
        self.robot_ip = robot_ip
        self.running = False
        self.ws_client = ws.Client(url=ws_url)
        self.stream_ws_client = ws.Client(url=stream_url) if stream_url else None

        self.video_stream = UbtechCameraVideoStream(
            frame_callback=self.ws_client.send_message,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            robot_ip=robot_ip,
        )

    def register_message_callback(self, callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : Optional[callable]
            The callback function to process VLM results.
        """
        if callback is not None:
            self.ws_client.register_message_callback(callback)

    def start(self):
        """
        Start the Ubtech VLM provider.
        """
        if self.running:
            logging.warning("Ubtech VLM provider already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("Ubtech VLM provider started")

    def stop(self):
        """
        Stop the Ubtech VLM provider.
        """
        self.running = False
        self.video_stream.stop()
        self.ws_client.stop()

        if self.stream_ws_client:
            self.stream_ws_client.stop()

        logging.info("Ubtech VLM provider stopped")
