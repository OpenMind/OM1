import logging
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import VideoZenohStream

from .singleton import singleton


@singleton
class VLMVilaZenohProvider:
    """
    VLM Provider that handles video zenoh communication.

    This class implements a singleton pattern to manage video stream from zenoh
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        ws_url: str,
        topic: str = "rgb_image",
        decode_format: str = "H264",
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        topic : str
            The zenoh topic for the video stream. Defaults to "rgb_image".
        decode_format : str
            The decode format for the video stream. Defaults to "H264".
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.video_stream: VideoZenohStream = VideoZenohStream(
            topic,
            decode_format,
            frame_callback=self.ws_client.send_message,
        )

    def register_frame_callback(self, video_callback: Optional[Callable]):
        """
        Register a callback for processing video frames.

        Parameters
        ----------
        video_callback : callable
            The callback function to process video frames.
        """
        if video_callback is not None:
            self.video_stream.register_frame_callback(video_callback)

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : callable
            The callback function to process VLM results.
        """
        if message_callback is not None:
            self.ws_client.register_message_callback(message_callback)

    def start(self):
        """
        Start the VLM Zenoh provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("VLM Zenoh provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        logging.info("Vila VLM Zenoh provider started")

    def stop(self):
        """
        Stop the VLM Zenoh provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False

        self.video_stream.stop()
        self.ws_client.stop()
