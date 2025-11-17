import logging
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import GazeboVideoStream

from .singleton import singleton


@singleton
class VLMVilaGazeboProvider:
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
        topic: str = "/camera",
        stream_url: Optional[str] = None,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        fps : int
            The fps for the VLM service connection.
        topic : str
            The Gazebo simulated camera topic for the VLM service connection.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: GazeboVideoStream = GazeboVideoStream(
            self.ws_client.send_message, fps=fps, topic=topic
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
            logging.warning("Vila Gazebo VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(self.stream_ws_client.send_message)

        logging.info("Vila Gazebo VLM provider started")

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
