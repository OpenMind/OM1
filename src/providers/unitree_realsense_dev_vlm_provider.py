import base64
import glob
import logging
import subprocess
import time
from typing import Callable, List, Optional, Tuple

import cv2
from om1_utils import ws
from om1_vlm import VideoStream

from .singleton import singleton

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class UnitreeRealSenseDevVideoStream(VideoStream):
    """
    Manages video capture and streaming from a camera device.

    Captures frames from a camera, encodes them to base64, and sends them
    through the callback. Supports both macOS and Linux devices.
    """

    def __init__(
        self,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
    ):
        """
        Initialize the video stream.

        Parameters
        ----------
        frame_callback : callable, optional
            A single callback function to process video frames. The callback
            receives a base64-encoded frame string as its argument.
        frame_callbacks : list of callables, optional
            A list of callback functions to process video frames. Each callback
            receives a base64-encoded frame string as its argument.
        fps : int, optional
            Frames per second for the video stream. Default is 30.
        resolution : tuple of int, optional
            The resolution for the video stream as (width, height). Default is (640, 480).
        jpeg_quality : int, optional
            The JPEG quality for frame encoding, ranging from 0 to 100. Default is 70.
        """
        super().__init__(
            frame_callback=frame_callback,
            frame_callbacks=frame_callbacks,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

    def on_video(self):
        """
        Main video capture and processing loop.

        Continuously captures frames from the camera, encodes them to base64,
        and sends them through the callback. If too many consecutive frame read
        errors occur, it will attempt to switch to another camera device.

        This method catches all exceptions so that failures do not affect other
        threads or processes.
        """
        tried_devices = set()
        try:
            # Find and open the initial camera device.
            current_cam = self._find_rgb_device(skip_devices=tried_devices)
            if current_cam is None:
                logger.error("No viable RGB camera found.")
                return
            tried_devices.add(current_cam)

            self._cap = self._open_camera(current_cam)
            while self._cap is None:
                # If opening the current camera failed, try the next one.
                current_cam = self._find_rgb_device(skip_devices=tried_devices)
                if current_cam is None:
                    logger.error("No viable camera devices found.")
                    return
                tried_devices.add(current_cam)
                self._cap = self._open_camera(current_cam)

            failure_count = 0
            max_retries = 30  # Maximum allowed consecutive read failures. This is after approximately 3 seconds.

            while self.running:
                if self._cap is None or not self._cap.isOpened():
                    logger.error("Video capture device is not opened.")
                    break

                ret, frame = self._cap.read()
                if not ret:
                    failure_count += 1
                    logger.error(
                        "Error reading frame from video stream (failure %d/%d)",
                        failure_count,
                        max_retries,
                    )
                    time.sleep(0.1)

                    if failure_count >= max_retries:
                        logger.error(
                            "Too many frame read errors. Trying another camera device."
                        )
                        self._cap.release()
                        new_cam = self._find_rgb_device(skip_devices=tried_devices)
                        if new_cam is None:
                            logger.error(
                                "No viable camera devices found. Exiting video capture loop."
                            )
                            break
                        tried_devices.add(new_cam)
                        self._cap = self._open_camera(new_cam)

                        # If the new device also fails to open, continue trying.
                        while self._cap is None:
                            new_cam = self._find_rgb_device(skip_devices=tried_devices)
                            if new_cam is None:
                                logger.error(
                                    "No viable camera devices found. Exiting video capture loop."
                                )
                                break
                            tried_devices.add(new_cam)
                            self._cap = self._open_camera(new_cam)
                        failure_count = (
                            0  # Reset failure counter after switching devices
                        )
                    continue  # Skip processing for this iteration

                failure_count = 0  # Reset on a successful read

                # Convert frame to base64, and catch any encoding errors.
                try:
                    _, buffer = cv2.imencode(".jpg", frame, self.encode_quality)
                    frame_data = base64.b64encode(buffer.tobytes()).decode("utf-8")
                except Exception as e:
                    logger.exception("Error encoding frame: %s", e)
                    continue

                if self.frame_callbacks:
                    for frame_callback in self.frame_callbacks:
                        frame_callback(frame_data)

                time.sleep(self.frame_delay)
        except Exception as e:
            logger.exception("Unhandled error in video stream: %s", e)
        finally:
            if self._cap:
                self._cap.release()
                logger.info("Released video capture device")

    def _open_camera(self, cam):
        """
        Attempt to open the camera device and set desired properties.

        Parameters
        ----------
        cam : str
            The device path (e.g., '/dev/video0')

        Returns
        -------
        cv2.VideoCapture or None
            The opened capture device, or None if it could not be opened.
        """
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            logger.error("Error opening video stream from %s", cam)
            return None
        try:
            if self.fps is not None:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            if self.resolution is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        except Exception as e:
            logger.exception(
                "Error setting camera properties for device %s: %s", cam, e
            )
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
        except Exception:
            pass
        return cap

    def _find_rgb_device(self, skip_devices=None):
        """
        Find the first viable RGB camera device supporting common RGB formats.
        Optionally skips devices listed in skip_devices.

        Parameters
        ----------
        skip_devices : set, optional
            A set of device paths to ignore.

        Returns
        -------
        str or None
            The first viable RGB device, or None if none is found.
        """
        if skip_devices is None:
            skip_devices = set()

        try:
            video_devices = sorted(glob.glob("/dev/video*"))
        except Exception as e:
            logger.exception("Failed to list video devices: %s", e)
            return None

        for device in video_devices:
            if device in skip_devices:
                continue

            try:
                result = subprocess.run(
                    ["v4l2-ctl", f"--device={device}", "--list-formats"],
                    capture_output=True,
                    text=True,
                    shell=False,
                )
                formats = result.stdout
            except Exception as e:
                logger.exception(
                    "Failed to run v4l2-ctl for device '%s': %s", device, e
                )
                continue

            try:
                if "MJPG" in formats or "YUYV" in formats:
                    logger.info(
                        "Found RGB device at %s with formats: %s", device, formats
                    )
                    return device
            except Exception as e:
                logger.exception(
                    "Error processing formats for device %s: %s", device, e
                )

        logger.warning("No RGB device found")
        return None


@singleton
class UnitreeRealSenseDevVLMProvider:
    """
    VLM Provider that handles audio streaming and websocket communication.

    This class implements a singleton pattern to manage audio input streaming and websocket
    communication for vlm services. It runs in a separate thread to handle
    continuous vlm processing.
    """

    def __init__(
        self,
        ws_url: str,
        fps: int = 15,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
        stream_url: Optional[str] = None,
    ):
        """
        Initialize the VLM Provider.

        Parameters
        ----------
        ws_url : str
            The websocket URL for the VLM service connection.
        fps : int, optional
            The fps for the VLM service connection. Default is 15.
        resolution : tuple of int, optional
            The resolution for the video stream. Default is (640, 480).
        jpeg_quality : int, optional
            The JPEG quality for the video stream. Default is 70.
        stream_url : str, optional
            The URL for the video stream. If not provided, defaults to None.
        """
        self.running: bool = False
        self.ws_client: ws.Client = ws.Client(url=ws_url)
        self.stream_ws_client: Optional[ws.Client] = (
            ws.Client(url=stream_url) if stream_url else None
        )
        self.video_stream: VideoStream = UnitreeRealSenseDevVideoStream(
            self.ws_client.send_message,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

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
        Start the VLM provider.

        Initializes and starts the websocket client, video stream, and processing thread
        if not already running.
        """
        if self.running:
            logging.warning("Unitree RealSenseDev VLM provider is already running")
            return

        self.running = True
        self.ws_client.start()
        self.video_stream.start()

        if self.stream_ws_client:
            self.stream_ws_client.start()
            self.video_stream.register_frame_callback(
                self.stream_ws_client.send_message
            )

        logging.info("Unitree RealSenseDev VLM provider started")

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
