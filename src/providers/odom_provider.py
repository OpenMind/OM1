import logging
import math
import multiprocessing as mp
import threading
import time
from enum import Enum
from queue import Empty
from typing import Any, Optional

import zenoh

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import (
    Odometry,
    PoseStamped,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    nav_msgs,
    open_zenoh_session,
)

from .singleton import singleton

# ----------------------------------------------------------------------
# Optional Unitree imports
# ----------------------------------------------------------------------

ChannelSubscriber: Optional[Any] = None
ChannelFactoryInitialize: Optional[Any] = None
PoseStamped_: Optional[Any] = None

try:
    from unitree.unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree.unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
except ImportError:
    logging.warning(
        "Unitree SDK or CycloneDDS not found. "
        "You only need this if connecting to a Unitree robot."
    )

rad_to_deg = 57.2958


class RobotState(Enum):
    """
    Enumeration for robot posture states.
    """

    STANDING = "standing"
    SITTING = "sitting"


# ----------------------------------------------------------------------
# Worker process
# ----------------------------------------------------------------------


def odom_processor(
    channel: str,
    data_queue: mp.Queue,
    URID: str = "",
    use_zenoh: bool = False,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Odom Provider.
    Runs in a separate process and forwards odom / pose data into a queue.
    """
    setup_logging("odom_processor", logging_config=logging_config)

    def zenoh_odom_handler(data: zenoh.Sample) -> None:
        try:
            odom: Odometry = nav_msgs.Odometry.deserialize(data.payload.to_bytes())
            data_queue.put(
                PoseWithCovarianceStamped(
                    header=odom.header,
                    pose=PoseWithCovariance(
                        pose=odom.pose.pose,
                        covariance=odom.pose.covariance,
                    ),
                )
            )
        except Exception as e:
            logging.error(f"Zenoh odom handler error: {e}")

    def pose_message_handler(data: PoseStamped_) -> None:  # type: ignore
        try:
            data_queue.put(data)  # type: ignore
        except Exception as e:
            logging.error(f"CycloneDDS pose handler error: {e}")

    if use_zenoh:
        if not URID:
            logging.warning("Aborting TurtleBot4 Navigation system, no URID provided")
            return

        try:
            session = open_zenoh_session()
            session.declare_subscriber(f"{URID}/c3/odom", zenoh_odom_handler)
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            return

    if not use_zenoh:
        if ChannelFactoryInitialize is None or ChannelSubscriber is None:
            logging.error("Unitree SDK not available — cannot initialize CycloneDDS")
            return

        try:
            ChannelFactoryInitialize(0, channel)  # type: ignore
            pose_subscriber = ChannelSubscriber(
                "rt/utlidar/robot_pose", PoseStamped_
            )  # type: ignore
            pose_subscriber.Init(pose_message_handler, 10)
        except Exception as e:
            logging.error(f"Error initializing CycloneDDS odom channel: {e}")
            return

    while True:
        time.sleep(0.1)


# ----------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------


@singleton
class OdomProvider:
    """
    Provides odometry and pose processing with safety validation
    and motion stability filtering.
    """

    def __init__(
        self,
        URID: str = "",
        use_zenoh: bool = False,
        channel: Optional[str] = "",
    ):
        logging.info("Booting Odom Provider")

        self.use_zenoh = use_zenoh
        self.URID = URID
        self.channel = channel

        self.data_queue: mp.Queue[PoseStamped] = mp.Queue()
        self._odom_reader_thread: Optional[mp.Process] = None
        self._odom_processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.body_height_cm = 0
        self.body_attitude: Optional[RobotState] = None

        self.moving = False
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

        self.start()

    def _is_invalid_number(self, v: float) -> bool:
        """
        Check whether a numeric value is NaN or infinite.
        """
        return math.isnan(v) or math.isinf(v)

    def start(self) -> None:
        """
        Start the odometry reader and processor threads.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Odom Provider is already running.")
            return

        if not self.channel and not self.use_zenoh:
            logging.error("Channel must be specified to start Odom Provider.")
            return

        self._odom_reader_thread = mp.Process(
            target=odom_processor,
            args=(
                self.channel,
                self.data_queue,
                self.URID,
                self.use_zenoh,
                get_logging_config(),
            ),
            daemon=True,
        )
        self._odom_reader_thread.start()

        self._odom_processor_thread = threading.Thread(
            target=self.process_odom, daemon=True
        )
        self._odom_processor_thread.start()

    def euler_from_quaternion(self, x: float, y: float, z: float, w: float):
        """
        Convert quaternion values to Euler angles (roll, pitch, yaw).
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def process_odom(self) -> None:
        """
        Main odometry processing loop.
        """
        MOVE_THRESHOLD = 0.012

        while not self._stop_event.is_set():
            try:
                pose_data = self.data_queue.get(timeout=1.0)
            except Empty:
                continue

            pose = pose_data.pose
            header = pose_data.header

            incoming_ts = header.stamp.sec + header.stamp.nanosec * 1e-9
            self.odom_rockchip_ts = max(self.odom_rockchip_ts, incoming_ts)
            self.odom_subscriber_ts = time.time()

            if any(
                self._is_invalid_number(v)
                for v in (pose.position.x, pose.position.y, pose.position.z)
            ):
                continue

            dx = pose.position.x - self.previous_x
            dy = pose.position.y - self.previous_y
            dz = pose.position.z - self.previous_z

            self.previous_x = pose.position.x
            self.previous_y = pose.position.y
            self.previous_z = pose.position.z

            delta = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.move_history = 0.7 * delta + 0.3 * self.move_history
            self.moving = delta > MOVE_THRESHOLD or self.move_history > MOVE_THRESHOLD

            yaw = self.euler_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )[2]

            self.odom_yaw_m180_p180 = round(yaw * rad_to_deg, 4)
            self.odom_yaw_0_360 = round((-yaw * rad_to_deg) % 360, 4)

            self.x = round(pose.position.x, 4)
            self.y = round(pose.position.y, 4)

    @property
    def position(self) -> dict:
        """
        Return the current odometry state.
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

    def stop(self) -> None:
        """
        Stop the odometry provider and all worker threads.
        """
        self._stop_event.set()

        if self._odom_reader_thread:
            self._odom_reader_thread.terminate()
            self._odom_reader_thread.join()

        if self._odom_processor_thread:
            self._odom_processor_thread.join()
