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

from .odom_provider_base import OdomProviderBase
from .singleton import singleton


def turtlebot4_odom_processor(
    data_queue: mp.Queue,
    URID: str,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the TurtleBot4 Odom Provider.
    This function runs in a separate process to periodically retrieve the odometry
    and pose data from the robot via Zenoh and put it into a multiprocessing queue.

    Parameters
    ----------
    data_queue : mp.Queue
        Queue for sending the retrieved odometry and pose data.
    URID : str
        The URID needed to connect to the Zenoh publisher in the local network.
    logging_config : LoggingConfig, optional
        Optional logging configuration. If provided, it will override the default logging settings.
    """
    setup_logging("turtlebot4_odom_processor", logging_config=logging_config)

    def zenoh_odom_handler(data: zenoh.Sample):
        """
        Zenoh handler for odometry data.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing the odometry data.
        """
        odom: Odometry = nav_msgs.Odometry.deserialize(data.payload.to_bytes())
        logging.debug(f"Zenoh odom handler: {odom}")

        data_queue.put(
            PoseWithCovarianceStamped(header=odom.header, pose=odom.pose.pose)  # type: ignore
        )

    if URID is None:
        logging.warning("Aborting TurtleBot4 Navigation system, no URID provided")
        return None
    else:
        logging.info(f"TurtleBot4 Navigation system is using URID: {URID}")

    try:
        session = open_zenoh_session()
        logging.info(f"Zenoh navigation provider opened {session}")
        logging.info(f"TurtleBot4 navigation listeners starting with URID: {URID}")
        session.declare_subscriber(f"{URID}/c3/odom", zenoh_odom_handler)
    except Exception as e:
        logging.error(f"Error opening Zenoh client: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class TurtleBot4OdomProvider(OdomProviderBase):
    """
    TurtleBot4 Odometry Provider.

    This class implements odometry management for TurtleBot4 robots using Zenoh
    for communication.
    """

    def __init__(self, URID: Optional[str] = None):
        """
        Initialize the TurtleBot4 Odom Provider.

        Parameters
        ----------
        URID : str, optional
            The URID needed to connect to the Zenoh publisher in the local network. If not provided, the provider will log a warning and not start the Zenoh subscriber.
        """
        super().__init__()
        self.URID = URID
        self.start()

    def start(self) -> None:
        """
        Start the TurtleBot4 Odom Provider.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("TurtleBot4 Odom Provider is already running.")
            return

        logging.info(f"Starting TurtleBot4 Odom Provider with URID: {self.URID}")

        self._odom_reader_thread = mp.Process(
            target=turtlebot4_odom_processor,
            args=(
                self.data_queue,
                self.URID,
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
