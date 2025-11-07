"""
VILA Vision Language Model integration for om1.

This module provides integration with the VILA (Vision Language) model
for real-time video analysis and description generation.
"""

from .args import VILAArgParser
from .video_stream_input import VideoStreamInput
from .vila_processor import VILAProcessor

__all__ = ["VILAProcessor", "VILAArgParser", "VideoStreamInput"]
