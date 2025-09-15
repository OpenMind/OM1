import logging
import sys
import time

import matplotlib.pyplot as plt

sys.path.insert(0, "../../src")

from zenoh_msgs import open_zenoh_session, sensor_msgs

logging.basicConfig(level=logging.INFO)


class Intel435ObstacleDector:
    def __init__(self):
        self.obstacle = []

        self.session = open_zenoh_session()

        self.session.declare_subscriber(
            "camera/realsense2_camera_node/depth/obstacle_point", self.obstacle_callback
        )

        logging.info("Zenoh is open for Intel435ObstacleDector")

    def obstacle_callback(self, msg):
        try:
            points = sensor_msgs.PointCloud.deserialize(msg.payload.to_bytes())
            obstacles = []
            for pt in points.points:
                x = pt.x
                y = pt.y
                z = pt.z
                obstacles.append({"x": x, "y": y, "z": z})
            self.obstacle = obstacles
        except Exception as e:
            logging.error(f"Error processing obstacle info: {e}")

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
