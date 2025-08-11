import logging
import math

import numpy as np
import zenoh

from zenoh_idl import sensor_msgs

from .singleton import singleton


@singleton
class D435Provider:
    """
    Provider for D435 camera data using Zenoh.

    Parameters:
    ----------
    camera_height: float = 0.45
        Height of the camera from the ground in meters.
    tilt_angle: float = 55
        Tilt angle of the camera in degrees.
    obstacle_threshold: float = 0.05
        Threshold for obstacle detection in meters (default is 5 cm).
    """

    def __init__(
        self,
        camera_height: float = 0.45,
        tilt_angle: float = 55,
        obstacle_threshold: float = 0.05,  # 5cm above ground
    ):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info = None
        self.camera_image = None

        self.camera_height = camera_height
        self.tilt_angle = tilt_angle
        self.obstacle_threshold = obstacle_threshold
        self.obstacle = []

        self.running = False

        self.session = zenoh.open(zenoh.Config())

        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/image_rect_raw", self.depth_callback
        )
        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/camera_info", self.depth_info_callback
        )

        logging.info("Zenoh is open for D435Provider")

        self.start()

    def depth_info_callback(self, msg: sensor_msgs.CameraInfo):
        """
        Callback for depth camera info messages.

        Parameters:
        -----------
        msg : sensor_msgs.CameraInfo
            The camera info message containing intrinsic parameters.
        """
        try:
            self.camera_info = sensor_msgs.CameraInfo.deserialize(
                msg.payload.to_bytes()
            )
            self.fx = self.camera_info.k[0]
            self.fy = self.camera_info.k[4]
            self.cx = self.camera_info.k[2]
            self.cy = self.camera_info.k[5]
        except Exception as e:
            logging.error(f"Error processing depth info: {e}")

    def image_to_world(
        self,
        u: int,
        v: int,
        depth_value: float,
    ) -> tuple:
        """
        Convert image coordinates to world coordinates.

        Parameters:
        -----------
        u : int
            The x-coordinate in the image.
        v : int
            The y-coordinate in the image.
        depth_value : float
            The depth value at the pixel (u, v).

        Returns:
        --------
        tuple
            World coordinates (x, y, z) or (None, None, None) if intrinsics are not available.
        """
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            logging.warning("Camera intrinsics not available yet")
            return None, None, None

        depth_meters = depth_value / 1000.0

        # Image to camera coordinates
        cam_x = (u - self.cx) * depth_meters / self.fx
        cam_y = (v - self.cy) * depth_meters / self.fy
        cam_z = depth_meters

        point_camera = np.array([cam_x, cam_y, cam_z])
        theta = np.radians(self.tilt_angle)

        R_tilt = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

        R_align = np.array(
            [
                [0, 0, 1],  # Camera Z (forward) -> World X (forward)
                [-1, 0, 0],  # Camera X (right) -> World Y (left)
                [0, -1, 0],  # Camera Y (down) -> World Z (up)
            ]
        )

        R_combined = R_align @ R_tilt
        point_world = R_combined @ point_camera

        camera_position_world = np.array([0, 0, self.camera_height])
        point_world = point_world + camera_position_world

        world_x = point_world[0]
        world_y = point_world[1]
        world_z = point_world[2]

        return world_x, world_y, world_z

    def imgmsg_to_numpy(self, img_msg: sensor_msgs.Image) -> np.ndarray:
        """
        Convert an image message to a NumPy array.

        Parameters:
        -----------
        img_msg : sensor_msgs.Image
            The image message to convert.

        Returns:
        --------
        np.ndarray
            The image as a NumPy array.
        """
        if img_msg.encoding == "mono8" or img_msg.encoding == "8UC1":
            dtype = np.uint8
        # Intel 435 supports 16-bit depth images
        elif img_msg.encoding == "mono16" or img_msg.encoding == "16UC1":
            dtype = np.uint16
        elif img_msg.encoding == "32FC1":
            dtype = np.float32
        elif img_msg.encoding == "bgr8" or img_msg.encoding == "rgb8":
            dtype = np.uint8
        else:
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

        data_bytes = bytes(img_msg.data)
        np_array = np.frombuffer(data_bytes, dtype=dtype)

        try:
            depth_image = np_array.reshape((img_msg.height, img_msg.width))
        except ValueError as e:
            logging.error(f"Error reshaping image data: {e}")
            return None

        return depth_image

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

    def depth_callback(self, msg: sensor_msgs.Image):
        """
        Callback for depth image messages.

        Parameters:
        -----------
        msg : sensor_msgs.Image
            The depth image message.
        """
        try:
            self.camera_image = sensor_msgs.Image.deserialize(msg.payload.to_bytes())

            depth_image = self.imgmsg_to_numpy(self.camera_image)
            if depth_image is None:
                logging.error("Failed to convert depth image")
                return

            obstacle = []

            for row in range(0, depth_image.shape[0], 10):
                for col in range(0, depth_image.shape[1], 10):
                    depth_value = depth_image[row, col]
                    if depth_value > 0:
                        world_x, world_y, world_z = self.image_to_world(
                            col, row, depth_value
                        )

                        if world_x is not None and world_z > self.obstacle_threshold:
                            angle_degrees, distance = self.calculate_angle_and_distance(
                                world_x, world_y
                            )
                            # Change to the robot coordinate system
                            obstacle.append(
                                {
                                    "x": -world_y,
                                    "y": world_x,
                                    "z": world_z,
                                    "depth": depth_value,
                                    "angle": angle_degrees,
                                    "distance": distance,
                                }
                            )

            self.obstacle = obstacle
            logging.debug(f"Detected {len(self.obstacle)} obstacles")

        except Exception as e:
            logging.error(f"Error processing depth image: {e}")

    def start(self):
        """
        Start the D435 provider.
        """

        if self.running:
            logging.info("D435Provider is already running")
            return

        self.running = True
