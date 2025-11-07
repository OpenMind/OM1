import argparse
import base64
import json
import logging
from typing import Callable, Optional

import cv2
import numpy as np

# nano_llm is only available on the Jetson devices
try:
    import nano_llm
    from jetson_utils import cudaAllocMapped, cudaConvertColor, cudaFromNumpy
    from nano_llm.plugins import VideoOutput
except ModuleNotFoundError:
    VideoOutput = None
    cudaFromNumpy = None
    cudaAllocMapped = None
    cudaConvertColor = None

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
        self.video_output: nano_llm.plugins.VideoOutput = None
        self.running: bool = True
        self.frame_callback: Optional[Callable] = None
        self.eos: bool = False

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
            try:
                frame_data = json.loads(message)
                frame_base64 = frame_data.get("frame", message)
            except json.JSONDecodeError:
                frame_base64 = message

            img_bytes = base64.b64decode(frame_base64)
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode image from WebSocket message")
                return

            # Convert OpenCV image (numpy array) to CUDA memory (in BGR format)
            bgr_cuda = cudaFromNumpy(frame, isBGR=True)

            # Allocate CUDA memory for RGB format
            rgb_cuda = cudaAllocMapped(
                width=bgr_cuda.width, height=bgr_cuda.height, format="rgb8"
            )

            # Convert from BGR to RGB
            cudaConvertColor(bgr_cuda, rgb_cuda)

            if self.frame_callback:
                # Process frame through VLM pipeline
                processed_frame = self.frame_callback(rgb_cuda)

                if processed_frame is not None and self.video_output:
                    self.video_output.process(processed_frame)

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

        # Initialize video output
        self.video_output = VideoOutput(args.rtp_url)
        self.video_output.start()

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
