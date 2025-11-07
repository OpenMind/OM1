import argparse
import logging
from typing import Any, Callable

# nano_llm is only available on the Jetson devices
try:
    from nano_llm.plugins import VideoOutput, VideoSource
except ModuleNotFoundError:
    VideoSource = None
    VideoOutput = None

from ..video import enumerate_video_devices

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VideoDeviceInput:
    """
    A class to manage video input and output devices, primarily designed for Jetson devices.

    This class provides functionality to initialize, manage, and control video input sources
    and output streams. It's specifically designed to work with NVIDIA Jetson devices and
    requires the nano_llm package for full functionality.
    """

    def __init__(self):
        self.video_source: Any = None
        self.video_output: Any = None

    def setup_video_devices(
        self,
        args: argparse.Namespace,
        callback: Callable[[bytes], None],
        cuda_stream: int = 0,
    ):
        """
        Set up video input and output devices with specified parameters.

        This method initializes both the video source (camera input) and video output
        (RTP stream). It automatically detects available video devices and configures
        them according to the provided parameters.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments containing RTP URL configuration.
            Expected to have 'rtp_url' attribute
        callback : Callable[[bytes], None]
            Function to process video frames.
            Takes raw frame data as bytes
        cuda_stream : int, optional
            CUDA stream identifier for GPU processing, by default 0

        Returns
        -------
        tuple[Any, Any]
            A tuple containing (video_source, video_output) instances

        Raises
        ------
        ModuleNotFoundError
            If nano_llm package is not available
        RuntimeError
            If video device initialization fails
        """
        devices = enumerate_video_devices()
        camindex = "/dev/video" + str(devices[0][0]) if devices else "/dev/video0"
        logger.info(f"Using camera: {camindex}")

        # Initialize video source and output
        self.video_source = VideoSource(camindex, cuda_stream=cuda_stream)
        self.video_source.add(callback, threaded=False)
        self.video_source.start()

        self.video_output = VideoOutput(args.rtp_url)
        self.video_output.start()

        return self.video_source, self.video_output

    def stop(self):
        """
        Stop and clean up video input and output devices.

        This method safely stops both the video source and output streams
        if they have been initialized. It should be called before destroying
        the VideoDeviceInput instance to ensure proper resource cleanup.
        """
        if self.video_source:
            self.video_source.stop()
        if self.video_output:
            self.video_output.stop()
