import logging
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import VideoStream

from .singleton import singleton


@singleton
class VLMVilaProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage audio input streaming and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        ws_url: str,
        fps: int = 30,
        stream_url: Optional[str] = None,
        camera_index: int = 0,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        fps : int
            The fps for the VLM service connection.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        camera_index : int
            The camera index for the video stream device. Defaults to 0.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: VideoStream = VideoStream(
            self.ws_client.send_message, fps=fps, device_index=camera_index
        )

    def register_frame_callback(self, video_callback: Optional[Callable]):
        """
        Register a callback for processing video frames.

        Parameters
        ----------
        video_callback : Optional[Callable]
            The callback function to process video frames.
        """
        if video_callback is not None:
            self.video_stream.register_frame_callback(video_callback)

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback for processing VLM results.

        Parameters
        ----------
        callback : Optional[Callable]
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
            logging.warning("VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(self.stream_ws_client.send_message)

        logging.info("Vila VLM provider started")

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
