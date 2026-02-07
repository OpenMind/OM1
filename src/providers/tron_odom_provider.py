import logging
import multiprocessing as mp
import threading
import time
from typing import Optional

import zenoh

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import (
    Odometry,
    PoseWithCovarianceStamped,
    nav_msgs,
    open_zenoh_session,
)

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton


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
class TronOdomProvider(OdomProviderBase):
    """
    Tron Odom Provider.

    This class implements odometry management for Tron robots using Zenoh
    for communication.

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
        super().__init__()
        self.topic = topic
        self.start()

    def start(self) -> None:
        """
        Start the Tron Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Tron Odom Provider is already running.")
            return

        if not self.topic:
            logging.error("Topic must be specified to start the Tron Odom Provider.")
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

    def _update_body_state(self, pose):
        """
        Update body height and attitude based on pose data for Tron robot.

        Parameters
        ----------
        pose : Pose
            The pose data containing position and orientation.
        """
        # Body height detection for Tron robot
        # Based on observed data:
        # - Sitting: z ≈ 0.55m (55cm)
        # - Standing: z ≈ 0.71m (71cm)
        self.body_height_cm = round(pose.position.z * 100.0)
        if self.body_height_cm > 60:
            self.body_attitude = RobotState.STANDING
        elif self.body_height_cm > 3:
            self.body_attitude = RobotState.SITTING
