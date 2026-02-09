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
    from unitree.unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
except ImportError:
    logging.warning(
        "Unitree SDK or CycloneDDS not found. You do not need this unless you are connecting to a Unitree robot."
    )

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton


def go2_odom_processor(
    channel: str,
    data_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Unitree Go2 Odom Provider.
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
    setup_logging("go2_odom_processor", logging_config=logging_config)

    def pose_message_handler(data: PoseStamped_):  # type: ignore
        """
        Handler for pose messages from CycloneDDS.

        Parameters
        ----------
        data : PoseStamped_
            The PoseStamped message containing the pose data.
        """
        logging.debug(f"Pose message handler: {data}")  # type: ignore
        data_queue.put(data)  # type: ignore

    try:
        ChannelFactoryInitialize(0, channel)  # type: ignore
    except Exception as e:
        logging.error(f"Error initializing Unitree Go2 odom channel: {e}")
        return

    try:
        pose_subscriber = ChannelSubscriber("rt/utlidar/robot_pose", PoseStamped_)  # type: ignore
        pose_subscriber.Init(pose_message_handler, 10)
        logging.info("CycloneDDS pose subscriber initialized successfully")
    except Exception as e:
        logging.error(f"Error opening CycloneDDS client: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class UnitreeGo2OdomProvider(OdomProviderBase):
    """
    Unitree Go2 Odometry Provider.

    This class implements odometry management for Unitree Go2 robots using CycloneDDS
    for communication.

    Parameters
    ----------
    channel : str
        The channel to connect to the robot, used for CycloneDDS.
    """

    def __init__(self, channel: Optional[str] = None):
        """
        Initialize the Unitree Go2 Odom Provider.

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
        Start the Unitree Go2 Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Go2 Odom Provider is already running.")
            return

        if not self.channel:
            logging.error("Channel must be specified to start the Go2 Odom Provider.")
            return

        logging.info(f"Starting Unitree Go2 Odom Provider on channel: {self.channel}")

        self._odom_reader_thread = mp.Process(
            target=go2_odom_processor,
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

    def _update_body_state(self, pose):
        """
        Update body height and attitude based on pose data for Unitree Go2.

        Parameters
        ----------
        pose : Pose
            The pose data containing position and orientation.
        """
        self.body_height_cm = round(pose.position.z * 100.0)
        if self.body_height_cm > 24:
            self.body_attitude = RobotState.STANDING
        elif self.body_height_cm > 3:
            self.body_attitude = RobotState.SITTING
