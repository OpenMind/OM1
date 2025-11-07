import argparse
import base64
import json
import logging
from typing import Any, Callable, Optional

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VideoStreamInput:
    """
    Handles WebSocket-based video streaming with CUDA-accelerated processing on Jetson devices.

    This class manages the reception of base64-encoded video frames via WebSocket,
    processes them using CUDA acceleration, and optionally streams the processed output
    via RTP. It's specifically designed for NVIDIA Jetson devices and requires the
    nano_llm package and Jetson Utils for full functionality.

    The class supports real-time video processing pipelines with custom frame callbacks
    and includes error handling for various failure modes.
    """

    def __init__(self):
        self.running: bool = True
        self.frame_callback: Optional[Callable] = None
        self.video_output: Any = None

    def handle_ws_incoming_message(self, connection_id: str, message: str):
        """
        Process incoming WebSocket messages containing base64-encoded video frames.

        Parameters
        ----------
        connection_id : str
            Unique identifier for the WebSocket connection
        message : str
            JSON string containing frame data and timestamp

        Raises
        ------
        Exception
            If frame processing fails at any stage
        """
        try:
            frame_data = json.loads(message)

            frame_base64 = frame_data.get("frame", message)

            img_bytes = base64.b64decode(frame_base64)

            if self.frame_callback:
                self.frame_callback(img_bytes)

        except json.JSONDecodeError:
            try:
                img_bytes = base64.b64decode(message)
                if self.frame_callback:
                    self.frame_callback(img_bytes)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_video_stream(
        self,
        args: argparse.Namespace,
        frame_callback: Optional[Callable],
        cuda_stream: int = 0,
    ):
        """
        Initialize video streaming components and callbacks.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments containing RTP URL configuration.
            Expected to have 'rtp_url' attribute
        frame_callback : Optional[Callable]
            Function to process video frames.
            Takes CUDA memory buffer as input
        cuda_stream : int, optional
            CUDA stream identifier for GPU processing, by default 0

        Returns
        -------
        tuple[VideoStreamInput, Any]
            A tuple containing (self, video_output)

        Raises
        ------
        ModuleNotFoundError
            If required Jetson packages are not available
        """
        logger.info("Initializing WebSocket video stream handler")

        self.frame_callback = frame_callback

        return self, self.video_output

    def register_frame_callback(
        self, frame_callback: Optional[Callable], threaded: bool = False
    ):
        """
        Register a callback function for frame processing.

        Parameters
        ----------
        frame_callback : Optional[Callable]
            Function to process video frames.
            Takes CUDA memory buffer as input
        threaded : bool, optional
            Flag for threaded processing.
            Currently not implemented.
            By default False
        """
        self.frame_callback = frame_callback

    def stop(self):
        """
        Stop video streaming and clean up resources.

        Stops the video output stream and sets the running flag to False.
        This method should be called before destroying the VideoStreamInput
        instance to ensure proper resource cleanup.
        """
        self.running = False

        if self.video_output:
            self.video_output.stop()
