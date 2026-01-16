import logging
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "../../src")

from zenoh_msgs import open_zenoh_session, sensor_msgs

logging.basicConfig(level=logging.INFO)


class Intel435ObstacleDector:
    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info = None
        self.camera_image = None

        self.obstacle_threshold = 0.05  # 5cm above ground
        self.obstacle = []

        self.running = False

        self.session = open_zenoh_session()

        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/image_rect_raw", self.depth_callback
        )
        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/camera_info", self.depth_info_callback
        )

        logging.info("Zenoh is open for Intel435ObstacleDector")

    def depth_info_callback(self, msg):
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

    def image_to_world(self, u, v, depth_value, camera_height=0.45, tilt_angle=55):
        """
        Convert image coordinates to world coordinates
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
        theta = np.radians(tilt_angle)

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

        camera_position_world = np.array([0, 0, camera_height])
        point_world = point_world + camera_position_world

        world_x = point_world[0]
        world_y = point_world[1]
        world_z = point_world[2]

        return world_x, world_y, world_z

    def imgmsg_to_numpy(self, img_msg):
        """Convert sensor_msgs/Image to numpy array directly"""
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

    def calculate_angle_and_distance(self, world_x, world_y):
        distance = math.sqrt(world_x**2 + world_y**2)

        angle_rad = math.atan2(world_y, world_x)
        angle_degrees = math.degrees(angle_rad)

        return angle_degrees, distance

    def depth_callback(self, msg):
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
                            col, row, depth_value, camera_height=0.45, tilt_angle=55
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

    def plot_obstacles(self):
        if len(self.obstacle) > 50:
            x_coords = [obs["x"] for obs in self.obstacle]
            y_coords = [obs["y"] for obs in self.obstacle]

            plt.clf()
            plt.scatter(x_coords, y_coords, c="red", alpha=0.6, s=10)
            plt.scatter(0, 0, c="blue", s=50, marker="o")
            plt.xlabel("X (forward) [m]")
            plt.ylabel("Y (left) [m]")
            plt.title(f"Obstacles Top View ({len(self.obstacle)} points)")
            plt.grid(True, alpha=0.3)
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.pause(0.001)
            plt.show(block=False)


def main():
    detector = Intel435ObstacleDector()
    try:
        while True:
            detector.plot_obstacles()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down Intel435ObstacleDector")


if __name__ == "__main__":
    main()
