import logging
import multiprocessing as mp
import threading
import time
from queue import Empty, Full
from typing import Optional

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import load_session_config, open_zenoh_session
from zenoh_msgs.idl.unitree_go import SportModeState

from .singleton import singleton

state_machine_codes = {
    100: "Agile",
    1001: "Damping",
    1002: "Standing Lock",
    1004: "Crouch",  # Also maps to 2006
    1006: "Greeting/Stretching/Dancing/Bowing/Heart Shape/Happy",
    1007: "Sit",
    1008: "Front Jump",
    1009: "Lunge",
    1013: "Balance Standing",
    1015: "Regular Walking",
    1016: "Regular Running",
    1017: "Regular Endurance",
    1091: "Strike a Pose",
    2006: "Crouch",  # Duplicate of 1004
    2007: "Dodge",
    2008: "Bound Run",
    2009: "Jump Run",
    2010: "Classic",
    2011: "Handstand",
    2012: "Front Flip",
    2013: "Back Flip",
    2014: "Left Flip",
    2016: "Cross Step",
    2017: "Upright",
    2019: "Towing",
}


def _state_zenoh_processor(
    api_key: Optional[str],
    use_sim: bool,
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Unitree Go2 state provider using Zenoh.

    Parameters
    ----------
    api_key : Optional[str]
        API key for authentication with the Zenoh broker, if required.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    data_queue : mp.Queue
        The multiprocessing queue to send decoded state data back to the main process.
    control_queue : mp.Queue
        The multiprocessing queue to receive control commands (e.g. to stop the processor).
    logging_config : LoggingConfig, optional
        The logging configuration to use for this processor. If None, default logging is used.
    """
    setup_logging("unitree_go2_state_zenoh_processor", logging_config=logging_config)

    def on_sample(sample) -> None:
        try:
            msg = SportModeState.deserialize(sample.payload.to_bytes())
        except Exception:
            logging.exception("failed to decode SportModeState on sportmodestate")
            return
        data = {
            "go2_sport_mode_state_msg": msg,
            "go2_state_code": msg.error_code,
            "go2_state": state_machine_codes.get(msg.error_code, "unknown"),
            "go2_action_progress": msg.progress,
        }
        try:
            data_queue.put_nowait(data)
        except Full:
            try:
                data_queue.get_nowait()
                data_queue.put_nowait(data)
            except Empty:
                pass

    try:
        load_session_config(api_key, use_sim)
        session = open_zenoh_session()

        session.declare_subscriber("sportmodestate", on_sample)
        logging.info("Subscribed to Unitree Go2 state topic: sportmodestate")
    except Exception:
        logging.exception("failed to open Zenoh session for sportmodestate")
        return

    while True:
        try:
            if control_queue.get_nowait() == "STOP":
                break
        except Empty:
            pass
        time.sleep(0.1)


@singleton
class UnitreeGo2StateZenohProvider:
    """
    Unitree Go2 State Provider.
    """

    def __init__(self, api_key: Optional[str] = None, use_sim: bool = False):
        """
        Initialize the Unitree Go2 State Provider, setting up the Zenoh subscription and internal state management.

        Parameters
        ----------
        api_key : Optional[str]
            API key for authentication with the Zenoh broker, if required.
        use_sim : bool
            Whether to use the simulation Zenoh endpoint instead of a local one.
        """
        self.api_key = api_key
        self.use_sim = use_sim

        self.data_queue: mp.Queue = mp.Queue(maxsize=5)
        self.control_queue: mp.Queue = mp.Queue()

        self._reader_proc: Optional[mp.Process] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.go2_sport_mode_state_msg = None
        self.go2_state: Optional[str] = None
        self.go2_state_code: Optional[int] = None
        self.go2_action_progress: int = 0

        self.start()

    def start(self) -> None:
        """
        Start the reader process and processor thread (idempotent).
        """
        if not self._reader_proc or not self._reader_proc.is_alive():
            self._reader_proc = mp.Process(
                target=_state_zenoh_processor,
                args=(self.api_key, self.use_sim, self.data_queue, self.control_queue, get_logging_config()),
                daemon=True,
            )
            self._reader_proc.start()
            logging.info("Unitree Go2 Zenoh state reader started.")

        if not self._processor_thread or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
            self._processor_thread.start()
            logging.info("Unitree Go2 Zenoh state processor started.")

    def stop(self) -> None:
        """
        Stop the reader process and processor thread.
        """
        self._stop_event.set()
        if self._reader_proc:
            self.control_queue.put("STOP")
            self._reader_proc.terminate()
            self._reader_proc.join(timeout=2)

        if self._processor_thread:
            self._processor_thread.join(timeout=2)

    def _processor_loop(self) -> None:
        """
        Process the Unitree Go2 state data from the data queue.
        """
        while not self._stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=0.5)
            except Empty:
                continue

            self.go2_sport_mode_state_msg = data.get("go2_sport_mode_state_msg")
            self.go2_state = data.get("go2_state")
            self.go2_state_code = data.get("go2_state_code")
            self.go2_action_progress = data.get("go2_action_progress")

    @property
    def state(self) -> Optional[str]:
        """
        Get the current state of the Unitree Go2 robot.

        Returns
        -------
        Optional[str]
            The current state of the robot, or None if not available.
        """
        return self.go2_state

    @property
    def state_code(self) -> Optional[int]:
        """
        Get the current state code of the Unitree Go2 robot.

        Returns
        -------
        Optional[int]
            The current state code of the robot, or None if not available.
        """
        return self.go2_state_code

    @property
    def action_progress(self) -> int:
        """
        Get the current action progress of the Unitree Go2 robot.

        Returns
        -------
        int
            The current action progress of the robot, or 0 if not in the action mode.
        """
        return self.go2_action_progress
