import math

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class Intel435ObstacleDector(Node):
    def __init__(self):
        super().__init__("intel435_obstacle_dector")
        self.bridge = CvBridge()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.obstacle_threshold = 0.10  # 10cm above ground
        self.obstacle = []

        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/realsense2_camera_node/depth/image_rect_raw",
            self.depth_callback,
            10,
        )

        self.depth_info = self.create_subscription(
            CameraInfo,
            "/camera/realsense2_camera_node/depth/camera_info",
            self.depth_info_callback,
            10,
        )

        self.plot_timer = self.create_timer(0.01, self.plot_obstacles)

        self.get_logger().info("Intel435ObstacleDector node started")

    def depth_info_callback(self, msg):
        try:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
        except Exception as e:
            self.get_logger().error(f"Error processing depth info: {e}")

    def image_to_world(self, u, v, depth_value, camera_height=0.45, tilt_angle=55):
        """
        Convert image coordinates to world coordinates
        """
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn("Camera intrinsics not available yet")
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

    def calculate_angle_and_distance(self, world_x, world_y):
        distance = math.sqrt(world_x**2 + world_y**2)

        angle_rad = math.atan2(world_y, world_x)
        angle_degrees = math.degrees(angle_rad)

        return angle_degrees, distance

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            obstacle = []

            for row in range(0, depth_image.shape[0], 10):
                for col in range(0, depth_image.shape[1], 10):
                    depth_value = depth_image[row, col]
                    if depth_value > 0 and depth_value < 5000:
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
            self.get_logger().debug(f"Detected {len(self.obstacle)} obstacles")

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

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


def main(args=None):
    rclpy.init(args=args)
    node = Intel435ObstacleDector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
