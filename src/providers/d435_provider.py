import logging
import math
import threading

import zenoh

from zenoh_msgs import open_zenoh_session, sensor_msgs

from .singleton import singleton


@singleton
class D435Provider:
    """
    Provider for D435 camera data using Zenoh.

    This class provides thread-safe access to obstacle point cloud data
    from a D435 depth camera via Zenoh messaging. The obstacle data is
    updated asynchronously through a callback mechanism and can be safely
    accessed from multiple threads.
    """

    def __init__(self):
        """
        Initialize the D435Provider with Zenoh session and thread lock.

        This method sets up the Zenoh subscriber for obstacle point cloud data
        and initializes thread-safe storage for obstacle information. The provider
        automatically starts after successful initialization.

        Notes
        -----
        - If Zenoh session initialization fails, the provider will still be
          created but will not receive obstacle data until the session is
          successfully established.
        - The provider uses a thread lock to ensure safe concurrent access
          to the obstacle data from multiple threads.
        """
        self._lock = threading.Lock()
        self._obstacle: list[dict[str, float]] = []
        self.running: bool = False
        self.session = None

        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                "camera/realsense2_camera_node/depth/obstacle_point",
                self.obstacle_callback,
            )
            logging.info("Zenoh is open for D435Provider")
        except Exception as e:
            logging.error(
                f"Error opening Zenoh client for D435Provider: {e}",
                exc_info=True,
            )

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

        This method is called asynchronously by the Zenoh subscriber when new
        obstacle point cloud data is received. It processes the point cloud,
        calculates angles and distances, and updates the obstacle list in a
        thread-safe manner.

        Parameters
        ----------
        sample : zenoh.Sample
            The sample containing the point cloud data.

        Notes
        -----
        - This callback runs in a separate thread, so all updates to shared
          state (self.obstacle) are protected by a thread lock.
        - If deserialization or processing fails, an error is logged with
          full exception context for debugging.
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
            with self._lock:
                self._obstacle = obstacles
        except Exception as e:
            logging.error(
                f"Error processing obstacle point cloud data in D435Provider: {e}",
                exc_info=True,
            )

    @property
    def obstacle(self):
        """
        Get the current obstacle point cloud data in a thread-safe manner.

        Returns
        -------
        list
            A list of dictionaries, each containing obstacle point information
            with keys: 'x', 'y', 'z', 'angle', 'distance'.
        """
        with self._lock:
            return list(self._obstacle)

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
