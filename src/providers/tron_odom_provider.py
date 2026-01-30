import logging
import math
import multiprocessing as mp
import threading
import time
from enum import Enum
from typing import Optional

import zenoh

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import (
    Odometry,
    PoseWithCovarianceStamped,
    nav_msgs,
    open_zenoh_session,
)

from .singleton import singleton

rad_to_deg = 57.2958


class RobotState(Enum):
    """
    Enumeration for robot states.
    """

    STANDING = "standing"
    SITTING = "sitting"


def tron_odom_processor(
    topic: str,
    data_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Tron Odom Provider.
    This function runs in a separate process to periodically retrieve the odometry
    data from the robot via Zenoh and put it into a multiprocessing queue.

    Parameters
    ----------
    topic : str
        The Zenoh topic to subscribe to for odometry data.
    data_queue : mp.Queue
        Queue for sending the retrieved odometry and pose data.
    logging_config : LoggingConfig, optional
        Optional logging configuration. If provided, it will override the default logging settings.
    """
    setup_logging("tron_odom_processor", logging_config=logging_config)

    def zenoh_odom_handler(data: zenoh.Sample):
        """
        Zenoh handler for odometry data.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing the odometry data.
        """
        odom: Odometry = nav_msgs.Odometry.deserialize(data.payload.to_bytes())
        logging.debug(f"Tron Zenoh odom handler: {odom}")

        data_queue.put(
            PoseWithCovarianceStamped(header=odom.header, pose=odom.pose)  # type: ignore
        )

    try:
        session = open_zenoh_session()
        logging.info(f"Tron Zenoh odom provider opened session: {session}")
        logging.info(f"Tron odom listener subscribing to topic: {topic}")
        session.declare_subscriber(topic, zenoh_odom_handler)
    except Exception as e:
        logging.error(f"Error opening Zenoh client for Tron odom: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class TronOdomProvider:
    """
    Tron Odom Provider.

    This class implements a singleton pattern to manage:
        * Odom data from the Tron robot using Zenoh

    Parameters
    ----------
    topic : str
        The Zenoh topic to subscribe to for odometry data.
        Defaults to "odom".
    """

    def __init__(self, topic: str = "odom"):
        """
        Initialize the Tron Odom Provider with Zenoh configuration.

        Parameters
        ----------
        topic : str
            The Zenoh topic to subscribe to for odometry data.
            Defaults to "odom".
        """
        logging.info("Booting Tron Odom Provider")

        self.topic = topic

        self.data_queue: mp.Queue[PoseWithCovarianceStamped] = mp.Queue()
        self._odom_reader_thread: Optional[mp.Process] = None
        self._odom_processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.body_height_cm = 0
        self.body_attitude: Optional[RobotState] = None

        self.moving: bool = False
        self.previous_x = 0
        self.previous_y = 0
        self.previous_z = 0
        self.move_history = 0

        self._odom: Optional[Odometry] = None

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom_yaw_m180_p180 = 0.0
        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0

        self.start()

    def start(self) -> None:
        """
        Start the Tron Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Tron Odom Provider is already running.")
            return
        else:
            if not self.topic:
                logging.error(
                    "Topic must be specified to start the Tron Odom Provider."
                )
                return

            logging.info(f"Starting Tron Odom Provider on Zenoh topic: {self.topic}")

            self._odom_reader_thread = mp.Process(
                target=tron_odom_processor,
                args=(
                    self.topic,
                    self.data_queue,
                    get_logging_config(),
                ),
                daemon=True,
            )
            self._odom_reader_thread.start()

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            logging.warning("Tron Odom processor thread is already running.")
            return
        else:
            logging.info("Starting Tron Odom processor thread")
            self._odom_processor_thread = threading.Thread(
                target=self.process_odom, daemon=True
            )
            self._odom_processor_thread.start()

    def euler_from_quaternion(self, x: float, y: float, z: float, w: float) -> tuple:
        """
        https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise).

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

        Parameters
        ----------
        pose : PoseWithCovariance
            The pose data containing position and orientation.
        """
        while not self._stop_event.is_set():
            try:
                pose_data = self.data_queue.get()
            except Exception as e:
                logging.error(f"Error getting pose from queue: {e}")
                time.sleep(1)
                continue

            pose = pose_data.pose.pose
            header = pose_data.header

            # this is the time according to the robot. It may be off by several seconds from
            # UTC
            self.odom_rockchip_ts = header.stamp.sec + header.stamp.nanosec * 1e-9

            # The local timestamp
            self.odom_subscriber_ts = time.time()

            # Body height detection for Tron robot
            # Based on observed data:
            # - Sitting: z ≈ 0.55m (55cm)
            # - Standing: z ≈ 0.71m (71cm)
            self.body_height_cm = round(pose.position.z * 100.0)
            if self.body_height_cm > 60:
                self.body_attitude = RobotState.STANDING
            elif self.body_height_cm > 3:
                self.body_attitude = RobotState.SITTING

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
                    f"delta moving (m): {round(delta,3)} {round(self.move_history,3)}"
                )
            else:
                # logging.info(
                #     f"delta moving (m): {round(delta,3)} {round(self.move_history,3)}"
                # )
                self.moving = False

            angles = self.euler_from_quaternion(x, y, z, w)

            # this is in the standard robot convention
            # yaw increases when you turn LEFT
            # (counter-clockwise rotation about the vertical axis
            self.odom_yaw_m180_p180 = round(angles[2] * rad_to_deg, 4)

            # we also provide a second data product, where
            # * yaw increases when you turn RIGHT (CW), and
            # * the range runs from 0 to 360 Deg
            flip = -1.0 * self.odom_yaw_m180_p180
            if flip < 0.0:
                flip = flip + 360.0

            self.odom_yaw_0_360 = round(flip, 4)

            # current position in world frame
            self.x = round(pose.position.x, 4)
            self.y = round(pose.position.y, 4)
            logging.debug(
                f"tron odom: X:{self.x} Y:{self.y} W:{self.odom_yaw_m180_p180} H:{self.odom_yaw_0_360} T:{self.odom_rockchip_ts}"
            )

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
            - x: The x coordinate of the robot in the world frame.
            - y: The y coordinate of the robot in the world frame.
            - moving: A boolean indicating if the robot is currently moving.
            - odom_yaw_0_360: The yaw angle of the robot in degrees, ranging from 0 to 360.
            - body_height_cm: The height of the robot's body in centimeters.
            - body_attitude: The current attitude of the robot (e.g., sitting or standing).
            - odom_rockchip_ts: The unix timestamp of the last odometry update. Provided by the Zenoh publisher.
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
        Stop the TronOdomProvider and clean up resources.
        """
        self._stop_event.set()

        if self._odom_reader_thread:
            self._odom_reader_thread.terminate()
            self._odom_reader_thread.join()
            logging.info("TronOdomProvider reader thread stopped.")

        if self._odom_processor_thread:
            self._odom_processor_thread.join()
            logging.info("TronOdomProvider processor thread stopped.")
