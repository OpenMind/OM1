import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class DepthObstacleDetector(Node):
    def __init__(self):
        super().__init__('depth_obstacle_detector')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribe to depth image topic
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        # Camera parameters (adjust these based on your camera setup)
        self.camera_height = 0.3  # meters - height of camera from ground
        self.camera_tilt_angle = 15.0  # degrees - downward tilt angle
        self.max_detection_distance = 5.0  # meters
        self.obstacle_threshold = 0.1  # meters - minimum height to consider as obstacle

        # Depth image parameters
        self.depth_scale = 0.001  # Convert from mm to meters (typical for depth cameras)

        # Latest processed data
        self.latest_obstacles = []
        self.latest_depth_image = None

        # Colorbar reference
        self.colorbar = None

        # Plotting setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.setup_plots()

        # Animation object (to prevent garbage collection)
        self.animation = None

        # ROS2 timer for updating plots
        self.plot_timer = self.create_timer(0.1, self.update_plots_callback)  # 10 Hz

        # Initialize plots
        self.init_plots()

        self.get_logger().info('Depth Obstacle Detector initialized')

    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Depth image plot
        self.ax1.set_title('Depth Image')
        self.ax1.set_xlabel('Pixels')
        self.ax1.set_ylabel('Pixels')

        # Obstacle map plot
        self.ax2.set_title('Obstacle Map (Top View)')
        self.ax2.set_xlabel('Distance Forward (m)')
        self.ax2.set_ylabel('Distance Left/Right (m)')
        self.ax2.set_xlim(0, self.max_detection_distance)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')

    def depth_callback(self, msg):
        """Process incoming depth images"""
        try:
            # Convert ROS image to OpenCV format
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            elif msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
                return

            # Convert to meters if needed
            if msg.encoding == '16UC1':
                depth_image = depth_image.astype(np.float32) * self.depth_scale

            # Store for visualization
            self.latest_depth_image = depth_image.copy()

            # Process obstacles
            obstacles = self.detect_obstacles(depth_image)
            self.latest_obstacles = obstacles

            self.get_logger().info(f'Detected {len(obstacles)} obstacle points')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def detect_obstacles(self, depth_image):
        """Detect obstacles from depth image"""
        obstacles = []
        height, width = depth_image.shape

        # Camera intrinsic parameters (adjust based on your camera)
        fx = width / 2  # Approximate focal length in pixels
        fy = height / 2
        cx = width / 2   # Principal point
        cy = height / 2

        # Convert camera tilt angle to radians
        tilt_rad = np.radians(self.camera_tilt_angle)

        # Process every nth pixel to reduce computation
        step = 5

        for v in range(0, height, step):
            for u in range(0, width, step):
                depth = depth_image[v, u]

                # Skip invalid depth values
                if depth <= 0 or depth > self.max_detection_distance:
                    continue

                # Convert pixel coordinates to 3D camera coordinates
                x_cam = (u - cx) * depth / fx
                y_cam = (v - cy) * depth / fy
                z_cam = depth

                # Transform to world coordinates considering camera tilt
                # Assuming camera is tilted down by tilt_angle from horizontal
                x_world = z_cam * np.cos(tilt_rad) - (y_cam + self.camera_height) * np.sin(tilt_rad)
                y_world = x_cam  # Left/right doesn't change
                z_world = z_cam * np.sin(tilt_rad) + (y_cam + self.camera_height) * np.cos(tilt_rad)

                # Check if point is above ground (obstacle)
                if z_world > self.obstacle_threshold and x_world > 0:
                    obstacles.append({
                        'x': x_world,
                        'y': y_world,
                        'z': z_world,
                        'pixel_u': u,
                        'pixel_v': v
                    })

        return obstacles

    def init_plots(self):
        """Initialize the plots"""
        plt.ion()  # Turn on interactive mode
        self.fig.show()

    def update_plots_callback(self):
        """Timer callback to update plots"""
        if not plt.fignum_exists(self.fig.number):
            # Figure was closed, stop the timer
            self.plot_timer.cancel()
            return

        self.update_plots()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plots(self):
        """Update the plots with latest data"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Remove existing colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        # Plot depth image
        if self.latest_depth_image is not None:
            # Normalize depth image for visualization
            depth_vis = np.clip(self.latest_depth_image, 0, self.max_detection_distance)
            depth_vis = (depth_vis / self.max_detection_distance * 255).astype(np.uint8)

            self.ax1.imshow(depth_vis, cmap='jet')
            self.ax1.set_title(f'Depth Image (0-{self.max_detection_distance}m)')
            self.ax1.set_xlabel('Pixels')
            self.ax1.set_ylabel('Pixels')

        # Plot obstacle map
        if self.latest_obstacles:
            x_coords = [obs['x'] for obs in self.latest_obstacles]
            y_coords = [obs['y'] for obs in self.latest_obstacles]
            z_coords = [obs['z'] for obs in self.latest_obstacles]

            # Color points by height
            scatter = self.ax2.scatter(x_coords, y_coords, c=z_coords,
                                    cmap='viridis', s=2, alpha=0.6)

            # Add colorbar only once
            if len(self.latest_obstacles) > 0:
                self.colorbar = plt.colorbar(scatter, ax=self.ax2, label='Height (m)')

        # Robot position (origin)
        self.ax2.plot(0, 0, 'ro', markersize=10, label='Robot')

        self.ax2.set_title(f'Obstacle Map - {len(self.latest_obstacles)} points')
        self.ax2.set_xlabel('Distance Forward (m)')
        self.ax2.set_ylabel('Distance Left/Right (m)')
        self.ax2.set_xlim(0, self.max_detection_distance)
        self.ax2.set_ylim(-3, 3)
        self.ax2.grid(True)
        self.ax2.legend()

        plt.tight_layout()

def main(args=None):
    rclpy.init(args=args)

    detector = DepthObstacleDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
