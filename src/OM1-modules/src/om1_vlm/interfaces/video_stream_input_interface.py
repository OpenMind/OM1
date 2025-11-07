import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple


class VideoStreamInputInterface(ABC):
    """Abstract base class for video stream input handlers."""

    @abstractmethod
    def __init__(self):
        """Initialize the video stream input handler."""
        self.video_output = None
        self.running: bool = True
        self.frame_callback: Optional[Callable] = None
        self.eos: bool = False

    @abstractmethod
    def handle_ws_incoming_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket messages containing video frames.

        Args:
            connection_id: Unique identifier for the connection
            message: Incoming message (expected to be base64 encoded image)
        """
        pass

    @abstractmethod
    def setup_video_stream(
        self,
        args: argparse.Namespace,
        frame_callback: Optional[Callable],
        cuda_stream: int = 0,
    ) -> Tuple[Any, Any]:
        """Set up the video stream with specified configuration.

        Args:
            args: Configuration arguments
            frame_callback: Callback function for frame processing
            cuda_stream: CUDA stream identifier

        Returns:
            Tuple of (self, video_output)
        """
        pass

    @abstractmethod
    def register_frame_callback(
        self, frame_callback: Optional[Callable], threaded: bool = False
    ):
        """Register a callback function for processing video frames.

        Args:
            frame_callback: Callback function that will be called for each frame
            threaded: Whether to run the callback in a separate thread
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the video stream."""
        pass
