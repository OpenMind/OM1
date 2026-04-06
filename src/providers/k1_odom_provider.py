import logging
import math
import multiprocessing as mp
import threading
import time
from typing import Optional

import zenoh

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import Odometer, open_zenoh_session

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton

rad_to_deg = 57.2958


def k1_odom_processor(
    topic: str,
    data_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the K1 Odom Provider.
    This function runs in a separate process to periodically retrieve the odometry
    data from the robot via Zenoh and put it into a multiprocessing queue.
    """
    setup_logging("k1_odom_processor", logging_config=logging_config)

    def zenoh_odom_handler(data: zenoh.Sample):
        """
        Zenoh handler for odometry data.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing the odometry data.
        """
        try:
            odom: Odometer = Odometer.deserialize(data.payload.to_bytes())
            logging.debug(f"K1 Zenoh odom handler: x={odom.x}, y={odom.y}, theta={odom.theta}")

            data_queue.put(
                {
                    "x": odom.x,
                    "y": odom.y,
                    "theta": odom.theta,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            logging.error(f"Error deserializing K1 odometry data: {e}")

    try:
        session = open_zenoh_session()
        logging.info(f"K1 Zenoh odom provider opened session: {session}")
        logging.info(f"K1 odom listener subscribing to topic: {topic}")
        session.declare_subscriber(topic, zenoh_odom_handler)
    except Exception as e:
        logging.error(f"Error opening Zenoh client for K1 odom: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class K1OdomProvider(OdomProviderBase):
    """
    K1 Odom Provider.

    This class implements odometry management for K1 robots using Zenoh
    for communication.

    Parameters
    ----------
    topic : str
        The Zenoh topic to subscribe to for odometry data.
        Defaults to "odometer_state".
    """

    def __init__(self, topic: str = "odometer_state"):
        """
        Initialize the K1 Odom Provider with Zenoh configuration.

        Parameters
        ----------
        topic : str
            The Zenoh topic to subscribe to for odometry data.
            Defaults to "odometer_state".
        """
        super().__init__()
        self.topic = topic
        self.start()

    def start(self) -> None:
        """
        Start the K1 Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("K1 Odom Provider is already running.")
            return

        if not self.topic:
            logging.error("Topic must be specified to start the K1 Odom Provider.")
            return

        logging.info(f"Starting K1 Odom Provider on Zenoh topic: {self.topic}")

        self._odom_reader_thread = mp.Process(
            target=k1_odom_processor,
            args=(
                self.topic,
                self.data_queue,
                get_logging_config(),
            ),
            daemon=True,
        )
        self._odom_reader_thread.start()

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            logging.warning("K1 Odom processor thread is already running.")
            return
        else:
            logging.info("Starting K1 Odom processor thread")
            self._odom_processor_thread = threading.Thread(target=self.process_odom, daemon=True)
            self._odom_processor_thread.start()

    def _update_body_state(self, pose):
        """
        Update body height and attitude based on pose data for K1 robot.

        Parameters
        ----------
        pose : Pose
            The pose data containing position and orientation.
        """
        # For K1 robot, we don't have z position data from the simple Odometer message
        # Assume robot is always standing when we receive odometry data
        self.body_attitude = RobotState.STANDING
        self.body_height_cm = 70  # Default standing height

    def process_odom(self):
        """
        Process the odom data from K1's custom Odometer message.
        This method runs in a separate thread and continuously processes
        odometry data from the queue.
        """
        while not self._stop_event.is_set():
            try:
                odom_data = self.data_queue.get(timeout=1)
            except Exception:
                # Queue timeout or other errors
                continue

            if not isinstance(odom_data, dict):
                logging.warning(f"Unexpected odom data type: {type(odom_data)}")
                continue

            # Extract data from custom Odometer message
            x = odom_data.get("x", 0.0)
            y = odom_data.get("y", 0.0)
            theta = odom_data.get("theta", 0.0)  # theta is in radians

            # Update timestamp
            self.odom_subscriber_ts = time.time()
            self.odom_rockchip_ts = odom_data.get("timestamp", self.odom_subscriber_ts)

            # Calculate movement delta
            dx = (x - self.previous_x) ** 2
            dy = (y - self.previous_y) ** 2

            self.previous_x = x
            self.previous_y = y

            delta = math.sqrt(dx + dy)

            # Moving? Use a decay kernel
            self.move_history = 0.7 * delta + 0.3 * self.move_history

            if delta > 0.01 or self.move_history > 0.01:
                self.moving = True
                logging.info(f"delta moving (m): {round(delta, 3)} {round(self.move_history, 3)}")
            else:
                self.moving = False

            # Convert theta (radians) to degrees
            # theta is already in standard robot convention where positive is counter-clockwise
            self.odom_yaw_m180_p180 = round(theta * rad_to_deg, 4)

            # Normalize to [-180, 180] range
            while self.odom_yaw_m180_p180 > 180.0:
                self.odom_yaw_m180_p180 -= 360.0
            while self.odom_yaw_m180_p180 < -180.0:
                self.odom_yaw_m180_p180 += 360.0

            # Provide alternate representation [0, 360] with clockwise positive
            flip = -1.0 * self.odom_yaw_m180_p180
            if flip < 0.0:
                flip = flip + 360.0

            self.odom_yaw_0_360 = round(flip, 4)

            # Current position in world frame
            self.x = round(x, 4)
            self.y = round(y, 4)

            # Update body state
            self._update_body_state(None)

            logging.debug(
                f"k1 odom: X:{self.x} Y:{self.y} Theta:{round(theta, 4)} "
                f"Yaw_m180_p180:{self.odom_yaw_m180_p180} Yaw_0_360:{self.odom_yaw_0_360}"
            )
