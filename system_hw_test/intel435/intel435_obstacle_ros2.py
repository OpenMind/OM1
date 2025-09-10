import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud


class Intel435ObstacleDector(Node):
    def __init__(self):
        super().__init__("intel435_obstacle_dector")
        self.obstacle = []

        self.depth_subscription = self.create_subscription(
            PointCloud,
            "/camera/realsense2_camera_node/depth/obstacle_point",
            self.obstacle_callback,
            10,
        )

        self.plot_timer = self.create_timer(0.01, self.plot_obstacles)

        self.get_logger().info("Intel435ObstacleDector node started")

    def obstacle_callback(self, msg: PointCloud):
        try:
            points = []
            for point in msg.points:
                x = point.x
                y = point.y
                z = point.z
                points.append({"x": x, "y": y, "z": z})
            self.obstacle = points

        except Exception as e:
            self.get_logger().error(f"Error processing depth info: {e}")

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
