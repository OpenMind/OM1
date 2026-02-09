import logging
from typing import Callable, Optional

from om1_utils import ws
from om1_vlm import VideoRTSPStream

from .singleton import singleton


@singleton
class VLMVilaRTSPProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage video stream from RTSP and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        ws_url: str,
        rtsp_url: str = "rtsp://localhost:8554/top_camera",
        decode_format: str = "H264",
        fps: int = 30,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        rtsp_url : str
            The RTSP URL for the video stream. Defaults to "rtsp://localhost:8554/top_camera".
        decode_format : str
            The decode format for the video stream. Defaults to "H264".
        fps : int
            The fps for the VLM service connection.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.video_stream: VideoRTSPStream = VideoRTSPStream(
            rtsp_url,
            decode_format,
            frame_callback=self.ws_client.send_message,
            fps=fps,
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
         message_callback : Optional[Callable]
            The callback function to process VLM results.
        """
        if message_callback is not None:
            self.ws_client.register_message_callback(message_callback)

    def start(self):
        """
        Start the VLM RTSP provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("VLM RTSP provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        logging.info("Vila VLM RTSP provider started")

    def stop(self):
        """
        Stop the VLM RTSP provider.

        Stops the websocket client, video stream, and processing thread.
        """
        self.running = False

        self.video_stream.stop()
        self.ws_client.stop()
