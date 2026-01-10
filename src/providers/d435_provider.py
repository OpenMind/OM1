import logging
import math

import zenoh

from zenoh_msgs import open_zenoh_session, sensor_msgs

from .prometheus_monitor import PrometheusMonitor
from .singleton import singleton


@singleton
class D435Provider:
    """
    Provider for D435 camera data using Zenoh.
    """

    def __init__(self):
        """
        Initialize the D435Provider instance.

        Sets up the Zenoh subscriber for obstacle point cloud data and starts the provider.
        """
        self.obstacle: list[dict[str, float]] = []
        self.running: bool = False
        self.session = None

        # Register with Prometheus monitor
        self._monitor = PrometheusMonitor()
        self._monitor.register(
            "D435Provider",
            metadata={"type": "depth_camera", "source": "zenoh"},
            recovery_callback=self._recover,
        )

        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                "camera/realsense2_camera_node/depth/obstacle_point",
                self.obstacle_callback,
            )
            logging.info("Zenoh is open for D435Provider")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self._monitor.report_error("D435Provider", str(e))

        self.start()

    def calculate_angle_and_distance(self, world_x: float, world_y: float) -> tuple:
        """
        Calculate the angle and distance from the world coordinates.

        Parameters
        ----------
        world_x : float
            The x-coordinate in the world.
        world_y : float
            The y-coordinate in the world.

        Returns
        -------
        tuple
            A tuple containing the angle in degrees and the distance.
        """
        distance = math.sqrt(world_x**2 + world_y**2)

        angle_rad = math.atan2(world_y, world_x)
        angle_degrees = math.degrees(angle_rad)

        return angle_degrees, distance

    def obstacle_callback(self, sample: zenoh.Sample):
        """
        Callback function to process the obstacle point cloud data.

        Parameters
        ----------
        sample : zenoh.Sample
            The sample containing the point cloud data.
        """
        try:
            points = sensor_msgs.PointCloud.deserialize(sample.payload.to_bytes())

            obstacles = []
            for pt in points.points:  # type: ignore
                x = pt.x
                y = pt.y
                z = pt.z
                angle, distance = self.calculate_angle_and_distance(x, y)
                obstacles.append(
                    {"x": x, "y": y, "z": z, "angle": angle, "distance": distance}
                )
            self.obstacle = obstacles
            self._monitor.heartbeat("D435Provider")
        except Exception as e:
            logging.error(f"Error processing obstacle info: {e}")
            self._monitor.report_error("D435Provider", str(e))

    def start(self):
        """
        Start the D435 provider.
        """
        if self.running:
            logging.info("D435Provider is already running")
            return

        self.running = True

    def stop(self):
        """
        Stop the D435 provider.
        """
        if not self.running:
            logging.info("D435Provider is not running")
            return

        self.running = False

        if self.session:
            self.session.close()

        logging.info("D435Provider stopped and Zenoh session closed")

    def _recover(self) -> bool:
        """
        Attempt to recover the D435 provider.

        Returns
        -------
        bool
            True if recovery was successful, False otherwise.
        """
        try:
            logging.info("D435Provider: Attempting recovery...")
            self.stop()
            self.start()
            logging.info("D435Provider: Recovery successful")
            return True
        except Exception as e:
            logging.error(f"D435Provider: Recovery failed: {e}")
            return False
