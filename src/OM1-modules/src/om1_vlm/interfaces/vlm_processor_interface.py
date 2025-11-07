import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class VLMProcessorInterface(ABC):
    """Abstract base class for VLM (Vision Language Model) processors."""

    @abstractmethod
    def __init__(
        self,
        model_args: argparse.Namespace,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the VLM processor.

        Args:
            model_args: Arguments for model configuration
            callback: Optional callback function for processing results
        """
        pass

    @abstractmethod
    def on_video(self, image: Any) -> Any:
        """Process a video frame.

        Args:
            image: Input image/frame to process

        Returns:
            Processed image/frame
        """
        pass

    @abstractmethod
    def process_frames(self, video_output: Any, video_source: Any):
        """Process video frames continuously.

        Args:
            video_output: Output video stream
            video_source: Input video source
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the processor."""
        pass
