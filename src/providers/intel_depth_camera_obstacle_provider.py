import logging
import threading
import time
from typing import Optional

import numpy as np
import pyrealsense2 as rs

from providers.singleton import singleton

logger = logging.getLogger(__name__)


@singleton
class IntelDepthCameraObstacleProvider:
    """Provider for Intel RealSense depth camera obstacle detection.

    This provider reads depth data from an Intel RealSense camera and detects
    obstacles for collision avoidance.
    """

    def __init__(
        self,
        depth_threshold: float = 0.3,  # Minimum safe distance in meters
        fps: int = 30,  # D435i supports 30fps at max resolution
        width: int = 1280,  # D435i max resolution
        height: int = 720,
    ):
        """Initialize the Intel depth camera obstacle provider.

        Args:
            depth_threshold: Minimum safe distance to obstacles in meters
            fps: Frames per second for the depth camera
            width: Width of the depth image
            height: Height of the depth image
        """
        self.depth_threshold = depth_threshold
        self.fps = fps
        self.width = width
        self.height = height

        # RealSense pipeline
        self.pipeline = None
        self.config = None
        self.depth_scale = 0.001  # Default value, will be updated from sensor

        # Thread management
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Obstacle detection results
        self._obstacle_detected = False
        self._min_distance = None
        self._depth_frame = None

        # Auto-start the provider
        self.start()

    def start(self):
        """Start the Intel depth camera obstacle provider."""
        if self._running:
            logger.warning("Intel depth camera obstacle provider already running")
            return

        try:
            # Configure RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Enable depth stream
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

            # Start pipeline
            profile = self.pipeline.start(self.config)

            # Get depth sensor
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"Depth scale: {self.depth_scale}")

            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

            logger.info("Intel depth camera obstacle provider started successfully")
        except Exception as e:
            logger.error(f"Failed to start Intel depth camera obstacle provider: {e}")
            self._running = False

    def stop(self):
        """Stop the Intel depth camera obstacle provider."""
        logger.info("Stopping Intel depth camera obstacle provider")
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

        logger.info("Intel depth camera obstacle provider stopped")

    def _run(self):
        """Main processing loop running in separate thread."""
        while self._running:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                # Get depth frame
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())

                # Process depth data
                self._process_depth_frame(depth_image)

                # Store frame for external access
                with self._lock:
                    self._depth_frame = depth_image

            except Exception as e:
                logger.error(f"Error in Intel depth processing loop: {e}")
                time.sleep(0.1)

    def _process_depth_frame(self, depth_image: np.ndarray):
        """Process depth frame to detect obstacles.

        Args:
            depth_image: Depth image as numpy array (values in mm)
        """
        # Convert depth values from mm to meters
        depth_m = depth_image * self.depth_scale

        # Filter valid depth values (ignore 0 values which mean no reading)
        valid_depths = depth_m[depth_m > 0]

        if len(valid_depths) > 0:
            # Get minimum distance
            min_distance = np.min(valid_depths)

            # Update detection status
            with self._lock:
                self._obstacle_detected = min_distance < self.depth_threshold
                self._min_distance = min_distance

            if self._obstacle_detected:
                logger.warning(
                    f"Obstacle detected by Intel Depth Camera at {min_distance:.2f}m"
                )
        else:
            with self._lock:
                self._obstacle_detected = False
                self._min_distance = None

    def obstacle_detected(self) -> bool:
        """Check if any obstacle is detected below the threshold.

        Returns:
            bool: True if obstacle detected, False otherwise
        """
        with self._lock:
            return self._obstacle_detected

    @property
    def min_distance(self) -> Optional[float]:
        """Get minimum distance to obstacle in meters."""
        with self._lock:
            return self._min_distance

    @property
    def is_running(self) -> bool:
        """Check if the provider is running."""
        return self._running

    @property
    def depth_frame(self) -> Optional[np.ndarray]:
        """Get the latest depth frame."""
        with self._lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None
