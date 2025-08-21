import logging
import math

import zenoh

from zenoh_idl import sensor_msgs

from .singleton import singleton


@singleton
class D435Provider:
    """
    Provider for D435 camera data using Zenoh.
    """

    def __init__(self):
        self.obstacle = []

        self.running = False

        self.session = zenoh.open(zenoh.Config())

        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/obstacle_point", self.obstacle_callback
        )

        logging.info("Zenoh is open for D435Provider")

        self.start()

    def calculate_angle_and_distance(self, world_x: float, world_y: float) -> tuple:
        """
        Calculate the angle and distance from the world coordinates.

        Parameters:
        -----------
        world_x : float
            The x-coordinate in the world.
        world_y : float
            The y-coordinate in the world.

        Returns:
        --------
        tuple
            A tuple containing the angle in degrees and the distance.
        """
        distance = math.sqrt(world_x**2 + world_y**2)

        angle_rad = math.atan2(world_y, world_x)
        angle_degrees = math.degrees(angle_rad)

        return angle_degrees, distance

    def obstacle_callback(self, msg: sensor_msgs.PointCloud):
        """
        Callback function to process the obstacle point cloud data.

        Parameters:
        ----------
        msg : sensor_msgs.PointCloud
            The message containing the point cloud data.
        """
        try:
            points = sensor_msgs.PointCloud.deserialize(msg.payload.to_bytes())

            obstacles = []
            for pt in points.points:
                x = pt.x
                y = pt.y
                z = pt.z
                angle, distance = self.calculate_angle_and_distance(x, y)
                obstacles.append(
                    {"x": x, "y": y, "z": z, "angle": angle, "distance": distance}
                )
            self.obstacle = obstacles
        except Exception as e:
            logging.error(f"Error processing obstacle info: {e}")

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
        self.session.close()
        logging.info("D435Provider stopped and Zenoh session closed")
