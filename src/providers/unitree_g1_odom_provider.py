import logging
import multiprocessing as mp
import threading
import time
from typing import Optional

from runtime.logging import LoggingConfig, get_logging_config, setup_logging

try:
    from unitree.unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
except ImportError:
    logging.warning(
        "Unitree SDK or CycloneDDS not found. You do not need this unless you are connecting to a Unitree robot."
    )

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton


def g1_odom_processor(
    channel: str,
    data_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Unitree G1 Odom Provider.
    This function runs in a separate process to periodically retrieve the odometry
    and pose data from the robot via CycloneDDS and put it into a multiprocessing queue.

    Parameters
    ----------
    channel : str
        The channel to connect to the robot.
    data_queue : mp.Queue
        Queue for sending the retrieved odometry and pose data.
    logging_config : LoggingConfig, optional
        Optional logging configuration. If provided, it will override the default logging settings.
    """
    setup_logging("g1_odom_processor", logging_config=logging_config)

    def sport_mode_handler(data: SportModeState_):  # type: ignore
        """
        Handler for SportModeState messages from CycloneDDS.

        Parameters
        ----------
        data : SportModeState_
            The SportModeState message containing odometry and IMU data.
        """
        logging.debug(f"SportModeState handler: {data}")  # type: ignore
        data_queue.put(data)  # type: ignore

    try:
        ChannelFactoryInitialize(0, channel)  # type: ignore
    except Exception as e:
        logging.error(f"Error initializing Unitree G1 odom channel: {e}")
        return

    try:
        sport_mode_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)  # type: ignore
        sport_mode_subscriber.Init(sport_mode_handler, 10)
        logging.info("CycloneDDS SportModeState subscriber initialized successfully")
    except Exception as e:
        logging.error(f"Error opening CycloneDDS client: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class UnitreeG1OdomProvider(OdomProviderBase):
    """
    Unitree G1 Odometry Provider.

    This class implements odometry management for Unitree G1 robots using CycloneDDS
    for communication.

    Parameters
    ----------
    channel : str
        The channel to connect to the robot, used for CycloneDDS.
    """

    def __init__(self, channel: Optional[str] = None):
        """
        Initialize the Unitree G1 Odom Provider.

        Parameters
        ----------
        channel : str
            The channel to connect to the robot, used for CycloneDDS.
        """
        super().__init__()
        self.channel = channel
        self.start()

    def start(self) -> None:
        """
        Start the Unitree G1 Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("G1 Odom Provider is already running.")
            return

        if not self.channel:
            logging.error("Channel must be specified to start the G1 Odom Provider.")
            return

        logging.info(f"Starting Unitree G1 Odom Provider on channel: {self.channel}")

        self._odom_reader_thread = mp.Process(
            target=g1_odom_processor,
            args=(
                self.channel,
                self.data_queue,
                get_logging_config(),
            ),
            daemon=True,
        )
        self._odom_reader_thread.start()

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            logging.warning("Odom processor thread is already running.")
            return
        else:
            logging.info("Starting Odom processor thread")
            self._odom_processor_thread = threading.Thread(
                target=self.process_odom, daemon=True
            )
            self._odom_processor_thread.start()

    def process_odom(self):
        """
        Process the G1 SportModeState data and update the internal state.
        This overrides the base class method to handle G1-specific message format.
        """
        import math

        from .odom_provider_base import rad_to_deg

        while not self._stop_event.is_set():
            try:
                sport_data = self.data_queue.get(timeout=1)
            except Exception:
                continue

            # Extract timestamp
            self.odom_rockchip_ts = (
                sport_data.stamp.sec + sport_data.stamp.nanosec * 1e-9
            )
            self.odom_subscriber_ts = time.time()

            # Extract position from the array [x, y, z]
            x_pos = sport_data.position[0]
            y_pos = sport_data.position[1]
            z_pos = sport_data.position[2]

            # Calculate movement delta
            dx = (x_pos - self.previous_x) ** 2
            dy = (y_pos - self.previous_y) ** 2
            dz = (z_pos - self.previous_z) ** 2

            self.previous_x = x_pos
            self.previous_y = y_pos
            self.previous_z = z_pos

            delta = math.sqrt(dx + dy + dz)

            # Update moving status with decay kernel
            self.move_history = 0.7 * delta + 0.3 * self.move_history

            if delta > 0.01 or self.move_history > 0.01:
                self.moving = True
                logging.info(
                    f"delta moving (m): {round(delta, 3)} {round(self.move_history, 3)}"
                )
            else:
                self.moving = False

            # Extract yaw directly from IMU RPY (roll, pitch, yaw)
            # rpy[2] is yaw in radians
            yaw_rad = sport_data.imu_state.rpy[2]

            # Convert to degrees (standard robot convention: yaw increases CCW)
            self.odom_yaw_m180_p180 = round(yaw_rad * rad_to_deg, 4)

            # Convert to 0-360 range (yaw increases CW)
            flip = -1.0 * self.odom_yaw_m180_p180
            if flip < 0.0:
                flip = flip + 360.0

            self.odom_yaw_0_360 = round(flip, 4)

            # Update current position
            self.x = round(x_pos, 4)
            self.y = round(y_pos, 4)
            self.z = round(z_pos, 4)

            # We assume that the robot is always standing
            self.body_attitude = RobotState.STANDING

            logging.debug(
                f"G1 odom: X:{self.x} Y:{self.y} Z:{self.z} W:{self.odom_yaw_m180_p180} H:{self.odom_yaw_0_360} T:{self.odom_rockchip_ts}"
            )
