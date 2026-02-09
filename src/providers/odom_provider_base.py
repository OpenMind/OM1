import logging
import math
import multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

rad_to_deg = 57.2958


class RobotState(Enum):
    """
    Enumeration for robot states.
    """

    STANDING = "standing"
    SITTING = "sitting"


class OdomProviderBase(ABC):
    """
    Base class for Odometry Providers.

    This abstract class provides common functionality for managing odometry
    and pose data across different robot platforms.
    """

    def __init__(self):
        """
        Initialize the base Odometry Provider.
        """
        logging.info(f"Booting {self.__class__.__name__}")

        self.data_queue: mp.Queue = mp.Queue()
        self._odom_reader_thread: Optional[mp.Process] = None
        self._odom_processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.body_height_cm = 0
        self.body_attitude: Optional[RobotState] = None

        self.moving: bool = False
        self.previous_x = 0.0
        self.previous_y = 0.0
        self.previous_z = 0.0
        self.move_history = 0.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom_yaw_m180_p180 = 0.0
        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0

    @abstractmethod
    def start(self) -> None:
        """
        Start the Odometry Provider.
        Must be implemented by subclasses.
        """
        pass

    def euler_from_quaternion(self, x: float, y: float, z: float, w: float) -> tuple:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise).

        https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/

        Parameters
        ----------
        x : float
            The x component of the quaternion.
        y : float
            The y component of the quaternion.
        z : float
            The z component of the quaternion.
        w : float
            The w component of the quaternion.

        Returns
        -------
        tuple
            A tuple containing the roll, pitch, and yaw angles in radians.
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def process_odom(self):
        """
        Process the odom data and update the internal state.
        This method runs in a separate thread and continuously processes
        odometry data from the queue.
        """
        while not self._stop_event.is_set():
            try:
                pose_data = self.data_queue.get(timeout=1)
            except Exception:
                # Queue timeout or other errors
                continue

            # Handle different message types
            # PoseWithCovarianceStamped has pose.pose, PoseStamped has just pose
            if hasattr(pose_data.pose, "pose"):
                # PoseWithCovarianceStamped
                pose = pose_data.pose.pose
            else:
                # PoseStamped or similar
                pose = pose_data.pose

            header = pose_data.header

            # This is the time according to the RockChip. It may be off by several seconds from UTC
            self.odom_rockchip_ts = header.stamp.sec + header.stamp.nanosec * 1e-9

            # The local timestamp
            self.odom_subscriber_ts = time.time()

            # Update body height and attitude if applicable
            self._update_body_state(pose)

            x = pose.orientation.x
            y = pose.orientation.y
            z = pose.orientation.z
            w = pose.orientation.w

            dx = (pose.position.x - self.previous_x) ** 2
            dy = (pose.position.y - self.previous_y) ** 2
            dz = (pose.position.z - self.previous_z) ** 2

            self.previous_x = pose.position.x
            self.previous_y = pose.position.y
            self.previous_z = pose.position.z

            delta = math.sqrt(dx + dy + dz)

            # moving? Use a decay kernel
            self.move_history = 0.7 * delta + 0.3 * self.move_history

            if delta > 0.01 or self.move_history > 0.01:
                self.moving = True
                logging.info(
                    f"delta moving (m): {round(delta, 3)} {round(self.move_history, 3)}"
                )
            else:
                self.moving = False

            angles = self.euler_from_quaternion(x, y, z, w)

            # This is in the standard robot convention
            # yaw increases when you turn LEFT
            # (counter-clockwise rotation about the vertical axis)
            self.odom_yaw_m180_p180 = round(angles[2] * rad_to_deg, 4)

            # We also provide a second data product, where
            # * yaw increases when you turn RIGHT (CW), and
            # * the range runs from 0 to 360 Deg
            flip = -1.0 * self.odom_yaw_m180_p180
            if flip < 0.0:
                flip = flip + 360.0

            self.odom_yaw_0_360 = round(flip, 4)

            # Current position in world frame
            self.x = round(pose.position.x, 4)
            self.y = round(pose.position.y, 4)
            logging.debug(
                f"odom: X:{self.x} Y:{self.y} W:{self.odom_yaw_m180_p180} H:{self.odom_yaw_0_360} T:{self.odom_rockchip_ts}"
            )

    def _update_body_state(self, pose):
        """
        Update body height and attitude based on pose data.
        Can be overridden by subclasses for robot-specific logic.

        Parameters
        ----------
        pose : Pose
            The pose data containing position and orientation.
        """
        # Default implementation does nothing
        # Subclasses can override this for robot-specific behavior
        pass

    @property
    def position(self) -> dict:
        """
        Get the current robot position in world frame.
        Returns a dictionary with x, y, and odom_yaw_0_360.

        Returns
        -------
        dict
            A dictionary containing the current position and orientation of the robot.
            Keys include:
            - odom_x: The x coordinate of the robot in the world frame.
            - odom_y: The y coordinate of the robot in the world frame.
            - moving: A boolean indicating if the robot is currently moving.
            - odom_yaw_0_360: The yaw angle of the robot in degrees, ranging from 0 to 360.
            - odom_yaw_m180_p180: The yaw angle of the robot in degrees, ranging from -180 to 180.
            - body_height_cm: The height of the robot's body in centimeters.
            - body_attitude: The current attitude of the robot (e.g., sitting or standing).
            - odom_rockchip_ts: The unix timestamp of the last odometry update. Provided by the publisher.
            - odom_subscriber_ts: The unix timestamp of the last odometry update according to the subscriber.
        """
        return {
            "odom_x": self.x,
            "odom_y": self.y,
            "moving": self.moving,
            "odom_yaw_0_360": self.odom_yaw_0_360,
            "odom_yaw_m180_p180": self.odom_yaw_m180_p180,
            "body_height_cm": self.body_height_cm,
            "body_attitude": self.body_attitude,
            "odom_rockchip_ts": self.odom_rockchip_ts,
            "odom_subscriber_ts": self.odom_subscriber_ts,
        }

    def stop(self):
        """
        Stop the OdomProvider and clean up resources.
        """
        self._stop_event.set()

        if self._odom_reader_thread:
            self._odom_reader_thread.terminate()
            self._odom_reader_thread.join()
            logging.info(f"{self.__class__.__name__} reader thread stopped.")

        if self._odom_processor_thread:
            self._odom_processor_thread.join()
            logging.info(f"{self.__class__.__name__} processor thread stopped.")
