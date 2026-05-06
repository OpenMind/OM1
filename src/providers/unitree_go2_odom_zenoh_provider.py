import logging
import multiprocessing as mp
import threading
from typing import Optional

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import ZenohSampleType, load_session_config, open_zenoh_session
from zenoh_msgs.idl.geometry_msgs import PoseStamped

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton


def _go2_odom_zenoh_processor(
    api_key: Optional[str],
    topic: str,
    use_sim: bool,
    data_queue: mp.Queue,
    logging_config: LoggingConfig | None = None,
) -> None:
    """
    Go2 odometry processor that runs in a separate process.

    Parameters
    ----------
    api_key : Optional[str]
        API key for authentication with the Zenoh broker, if required.
    topic : str
        The Zenoh topic to subscribe to for odometry data.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    data_queue : mp.Queue
        The multiprocessing queue to send decoded odometry data back to the main process.
    logging_config : LoggingConfig, optional
        The logging configuration to use for this process. If None, default logging is used.
    """
    setup_logging("go2_odom_zenoh_processor", logging_config=logging_config)

    def on_sample(sample: ZenohSampleType) -> None:
        try:
            msg = PoseStamped.deserialize(sample.payload.to_bytes())
        except Exception:
            logging.exception(f"failed to decode PoseStamped on {topic}")
            return
        data_queue.put(msg)

    try:
        load_session_config(api_key, use_sim)
        session = open_zenoh_session()
    except Exception:
        logging.exception("failed to open Zenoh session for go2 odom")
        return

    session.declare_subscriber(topic, on_sample)
    logging.info(f"Subscribed to Go2 odometry topic: {topic}")

    threading.Event().wait()


@singleton
class UnitreeGo2OdomZenohProvider(OdomProviderBase):
    """
    Zenoh-based odometry provider for Unitree Go2.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        topic: str = "utlidar/robot_pose",
        use_sim: bool = False,
    ) -> None:
        """
        Init the provider and start the background odometry subscriber process.

        Parameters
        ----------
        api_key : Optional[str]
            API key for authentication with the Zenoh broker, if required.
        topic : str
            The Zenoh topic to subscribe to for odometry data.
        schema : str
            The message schema for the odometry data, e.g. "geometry_msgs/msg/PoseStamped".
        use_sim : bool
            Whether to use the simulation Zenoh endpoint instead of a local one.
        """
        super().__init__()

        self.api_key = api_key
        self.topic = topic
        self.use_sim = use_sim

        self.start()

    def start(self) -> None:
        """
        Start the background odom subscriber thread.
        """
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Go2 Zenoh Odom Provider is already running.")
            return

        self._odom_reader_thread = mp.Process(
            target=_go2_odom_zenoh_processor,
            args=(self.api_key, self.topic, self.use_sim, self.data_queue, get_logging_config()),
            daemon=True,
        )
        self._odom_reader_thread.start()

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            return

        self._odom_processor_thread = threading.Thread(target=self.process_odom, daemon=True)
        self._odom_processor_thread.start()

    def _update_body_state(self, pose) -> None:  # type: ignore[no-untyped-def]
        """
        Update the robot's body state (standing/sitting) based on the z position.

        Parameters
        ----------
        pose : geometry_msgs/msg/PoseStamped
            The pose message containing the robot's position.
        """
        self.body_height_cm = round(pose.position.z * 100.0)
        if self.body_height_cm > 24:
            self.body_attitude = RobotState.STANDING
        elif self.body_height_cm > 3:
            self.body_attitude = RobotState.SITTING
