import logging
import math
import multiprocessing as mp
import threading
import time
from enum import Enum
from queue import Empty
from typing import Any, Optional, Union

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
# Unitree opsiyonel import – unbound olmaması için güvenli tanım
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
    STANDING = "standing"
    SITTING = "sitting"


# ----------------------------------------------------------------------
# Worker Process
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
    Runs in a separate process and forwards odom / pose into a queue.
    """

    setup_logging("odom_processor", logging_config=logging_config)

    def zenoh_odom_handler(data: zenoh.Sample):
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

    def pose_message_handler(data: PoseStamped_):  # type: ignore
        try:
            data_queue.put(data)  # type: ignore
        except Exception as e:
            logging.error(f"CycloneDDS pose handler error: {e}")

    # -------- Turtlebot / Zenoh --------

    if use_zenoh:
        if not URID:
            logging.warning("Aborting TurtleBot4 Navigation system, no URID provided")
            return

        logging.info(f"TurtleBot4 Navigation system is using URID: {URID}")

        try:
            session = open_zenoh_session()
            logging.info(f"Zenoh navigation provider opened {session}")
            session.declare_subscriber(f"{URID}/c3/odom", zenoh_odom_handler)
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            return

    # -------- Unitree / CycloneDDS --------

    if not use_zenoh:
        if ChannelFactoryInitialize is None or ChannelSubscriber is None:
            logging.error(
                "Unitree SDK not available — cannot initialize CycloneDDS odom channel"
            )
            return

        try:
            ChannelFactoryInitialize(0, channel)  # type: ignore
        except Exception as e:
            logging.error(f"Error initializing Unitree Go2 odom channel: {e}")
            return

        try:
            pose_subscriber = ChannelSubscriber(
                "rt/utlidar/robot_pose", PoseStamped_
            )  # type: ignore
            pose_subscriber.Init(pose_message_handler, 10)
            logging.info("CycloneDDS pose subscriber initialized successfully")
        except Exception as e:
            logging.error(f"Error opening CycloneDDS client: {e}")
            return

    while True:
        time.sleep(0.1)


# ----------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------


@singleton
class OdomProvider:
    """
    Provides odom + pose processing with motion stability filtering & safety guards.
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

        self.moving: bool = False
        self.previous_x = 0
        self.previous_y = 0
        self.previous_z = 0
        self.move_history = 0

        self._odom: Optional[Union[Odometry, PoseStamped_]] = None  # type: ignore

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom_yaw_m180_p180 = 0.0
        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0

        self.start()

    # ---------- safety helpers ----------

    def _is_invalid_number(self, v: float) -> bool:
        return math.isnan(v) or math.isinf(v)

    # ---------- startup ----------

    def start(self) -> None:
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Odom Provider is already running.")
            return

        if not self.channel and not self.use_zenoh:
            logging.error("Channel must be specified to start the Odom Provider.")
            return

        logging.info(f"Starting Odom Provider on channel: {self.channel}")

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

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            logging.warning("Odom processor thread is already running.")
            return

        self._odom_processor_thread = threading.Thread(
            target=self.process_odom, daemon=True
        )
        self._odom_processor_thread.start()

    # ---------- math ----------

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, +1.0), -1.0)
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    # ---------- main loop ----------

    def process_odom(self):

        MOVE_THRESHOLD = 0.012

        while not self._stop_event.is_set():
            try:
                pose_data = self.data_queue.get(timeout=1.0)
            except Empty:  # <-- Pylance-safe
                continue
            except Exception as e:
                logging.error(f"OdomProvider: queue error: {e}")
                time.sleep(0.2)
                continue

            pose = pose_data.pose
            header = pose_data.header

            # ---------- timestamp monotonicity ----------

            incoming_ts = header.stamp.sec + header.stamp.nanosec * 1e-9

            if self.odom_rockchip_ts and incoming_ts < self.odom_rockchip_ts:
                logging.warning(
                    "OdomProvider: non-monotonic odom timestamp — keeping last value"
                )
            else:
                self.odom_rockchip_ts = incoming_ts

            self.odom_subscriber_ts = time.time()

            # ---------- posture detection (Unitree only) ----------

            if self.channel and not self.use_zenoh:
                self.body_height_cm = round(pose.position.z * 100.0)

                if self.body_height_cm > 24:
                    self.body_attitude = RobotState.STANDING
                elif self.body_height_cm > 3:
                    self.body_attitude = RobotState.SITTING

            # ---------- SAFETY VALIDATION GUARDS ----------

            if any(
                self._is_invalid_number(v)
                for v in [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]
            ):
                logging.warning(
                    "OdomProvider: received invalid position values — frame skipped"
                )
                continue

            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]

            if any(self._is_invalid_number(q) for q in quat):
                logging.warning(
                    "OdomProvider: received invalid quaternion — frame skipped"
                )
                continue

            norm = math.sqrt(sum(q * q for q in quat))

            if norm != 0:
                quat = [q / norm for q in quat]
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ) = quat
            else:
                logging.warning(
                    "OdomProvider: quaternion norm = 0 — using last valid orientation"
                )

            # ---------- motion stability filter ----------

            dx = (pose.position.x - self.previous_x) ** 2
            dy = (pose.position.y - self.previous_y) ** 2
            dz = (pose.position.z - self.previous_z) ** 2

            self.previous_x = pose.position.x
            self.previous_y = pose.position.y
            self.previous_z = pose.position.z

            delta = math.sqrt(dx + dy + dz)

            if delta < 1e-6:
                delta = 0.0

            self.move_history = max(0.0, 0.7 * delta + 0.3 * self.move_history)

            self.moving = delta > MOVE_THRESHOLD or self.move_history > MOVE_THRESHOLD

            # ---------- orientation ----------

            angles = self.euler_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )

            self.odom_yaw_m180_p180 = round(angles[2] * rad_to_deg, 4)

            flip = -1.0 * self.odom_yaw_m180_p180
            if flip < 0.0:
                flip += 360.0

            self.odom_yaw_0_360 = round(flip, 4)

            self.x = round(pose.position.x, 4)
            self.y = round(pose.position.y, 4)

            logging.debug(
                f"odom: X:{self.x} Y:{self.y} "
                f"W:{self.odom_yaw_m180_p180} "
                f"H:{self.odom_yaw_0_360} "
                f"T:{self.odom_rockchip_ts}"
            )

    # ---------- public api ----------

    @property
    def position(self) -> dict:
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
        self._stop_event.set()

        if self._odom_reader_thread:
            self._odom_reader_thread.terminate()
            self._odom_reader_thread.join()
            logging.info("OdomProvider reader thread stopped.")

        if self._odom_processor_thread:
            self._odom_processor_thread.join()
            logging.info("OdomProvider processor thread stopped.")
