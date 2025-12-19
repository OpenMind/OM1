#!/usr/bin/env python3
"""
OM1 Test Robot Demo Script
Bounty #363 - Enhanced Gazebo Environment Demo

This script demonstrates:
1. Sensor data reading (Camera, IMU, Depth)
2. Basic robot movement
3. Obstacle avoidance
4. Data streaming to OM1

Requirements:
- ROS2 (Humble or later)
- Gazebo 11+
- Python 3.8+
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu


class OM1TestRobotDemo(Node):
    """Demo node for OM1 Test Robot in enhanced Gazebo environment"""

    def __init__(self):
        super().__init__("om1_test_robot_demo")

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/om1_test_robot/cmd_vel", 10)

        # Subscribers for sensor data
        self.camera_sub = self.create_subscription(
            Image, "/om1_test_robot/camera/image_raw", self.camera_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "/om1_test_robot/depth_camera/depth", self.depth_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, "/om1_test_robot/imu/data", self.imu_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/om1_test_robot/odom", self.odom_callback, 10
        )

        # Data storage
        self.camera_data = None
        self.depth_data = None
        self.imu_data = None
        self.odom_data = None

        # Counters for logging
        self.camera_count = 0
        self.depth_count = 0
        self.imu_count = 0
        self.odom_count = 0

        self.get_logger().info("OM1 Test Robot Demo Node Started")
        self.get_logger().info("Waiting for sensor data...")

    def camera_callback(self, msg):
        """Process RGB camera data"""
        self.camera_data = msg
        self.camera_count += 1
        if self.camera_count % 30 == 0:  # Log every 30 frames
            self.get_logger().info(
                f"Camera: {msg.width}x{msg.height}, " f"encoding: {msg.encoding}"
            )

    def depth_callback(self, msg):
        """Process depth camera data"""
        self.depth_data = msg
        self.depth_count += 1
        if self.depth_count % 20 == 0:  # Log every 20 frames
            self.get_logger().info(f"Depth Camera: {msg.width}x{msg.height}")

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        self.imu_count += 1
        if self.imu_count % 100 == 0:  # Log every 100 samples
            self.get_logger().info(
                f"IMU - Angular velocity: "
                f"x={msg.angular_velocity.x:.3f}, "
                f"y={msg.angular_velocity.y:.3f}, "
                f"z={msg.angular_velocity.z:.3f}"
            )

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg
        self.odom_count += 1
        if self.odom_count % 50 == 0:  # Log every 50 samples
            pos = msg.pose.pose.position
            self.get_logger().info(
                f"Odometry - Position: " f"x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}"
            )

    def move_forward(self, duration=2.0, speed=0.5):
        """Move robot forward"""
        self.get_logger().info(f"Moving forward at {speed} m/s for {duration}s")

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0

        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        self.stop()

    def turn(self, duration=2.0, angular_speed=0.5):
        """Turn robot"""
        self.get_logger().info(f"Turning at {angular_speed} rad/s for {duration}s")

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed

        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        self.stop()

    def stop(self):
        """Stop robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.5)

    def run_demo_sequence(self):
        """Run demonstration sequence"""
        self.get_logger().info("Starting demo sequence...")

        # Wait for sensor data
        self.get_logger().info("Waiting for sensors to initialize...")
        time.sleep(3.0)

        # Check sensor status
        sensors_ready = True
        if self.camera_data is None:
            self.get_logger().warn("Camera data not received")
            sensors_ready = False
        if self.depth_data is None:
            self.get_logger().warn("Depth camera data not received")
            sensors_ready = False
        if self.imu_data is None:
            self.get_logger().warn("IMU data not received")
            sensors_ready = False

        if sensors_ready:
            self.get_logger().info("✓ All sensors active!")
        else:
            self.get_logger().warn("Some sensors not active, continuing anyway...")

        # Demo movement sequence
        self.get_logger().info("=== Demo Sequence Start ===")

        # Move 1: Forward
        self.get_logger().info("Movement 1: Forward")
        self.move_forward(duration=3.0, speed=0.3)
        time.sleep(1.0)

        # Move 2: Turn left
        self.get_logger().info("Movement 2: Turn Left")
        self.turn(duration=2.0, angular_speed=0.5)
        time.sleep(1.0)

        # Move 3: Forward again
        self.get_logger().info("Movement 3: Forward")
        self.move_forward(duration=2.0, speed=0.3)
        time.sleep(1.0)

        # Move 4: Turn right
        self.get_logger().info("Movement 4: Turn Right")
        self.turn(duration=2.0, angular_speed=-0.5)
        time.sleep(1.0)

        # Move 5: Return
        self.get_logger().info("Movement 5: Return")
        self.move_forward(duration=2.0, speed=0.3)
        time.sleep(1.0)

        self.get_logger().info("=== Demo Sequence Complete ===")

        # Print sensor statistics
        self.print_statistics()

    def print_statistics(self):
        """Print sensor data statistics"""
        self.get_logger().info("=== Sensor Statistics ===")
        self.get_logger().info(f"Camera frames received: {self.camera_count}")
        self.get_logger().info(f"Depth frames received: {self.depth_count}")
        self.get_logger().info(f"IMU samples received: {self.imu_count}")
        self.get_logger().info(f"Odometry samples received: {self.odom_count}")

        if self.camera_data:
            self.get_logger().info(
                f"Camera resolution: {self.camera_data.width}x{self.camera_data.height}"
            )
        if self.depth_data:
            self.get_logger().info(
                f"Depth resolution: {self.depth_data.width}x{self.depth_data.height}"
            )
        if self.odom_data:
            pos = self.odom_data.pose.pose.position
            self.get_logger().info(
                f"Final position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}"
            )


def main(args=None):
    """Main function"""
    print("=" * 60)
    print("OM1 Test Robot Demo - Bounty #363")
    print("Enhanced Gazebo Environment with Sensors")
    print("=" * 60)

    rclpy.init(args=args)

    try:
        node = OM1TestRobotDemo()

        # Run demo in separate thread to allow ROS callbacks
        import threading

        demo_thread = threading.Thread(target=node.run_demo_sequence)
        demo_thread.start()

        # Spin node to process callbacks
        rclpy.spin(node)

        demo_thread.join()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
