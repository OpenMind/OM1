import logging
import multiprocessing as mp
import threading
import time
from queue import Empty, Full
from typing import Optional

from runtime.logging import LoggingConfig, get_logging_config, setup_logging

try:
    from unitree.unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
except ImportError:
    logging.error(
        "Unitree SDK or CycloneDDS not found. Please install the unitree_sdk2py package or CycloneDDS."
    )

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


def go2_state_processor(
    channel: str,
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Process Unitree Go2 state data from the CycloneDDS session.

    Parameters
    ----------
    channel : str
        CycloneDDS channel to subscribe to for Unitree Go2 state data.
    data_queue : mp.Queue
        Queue for receiving state data.
    control_queue : mp.Queue
        Queue for sending control commands.
    logging_config : Optional[LoggingConfig]
        Logging configuration.
    """
    setup_logging("unitree_go2_state_processor", logging_config=logging_config)

    def get_state_from_code(code: int) -> Optional[str]:
        """
        Get the state name from the state code.

        Parameters
        ----------
        code : int
            The state code.

        Returns
        -------
        str
            The state name corresponding to the code, or "unknown" if not found.
        """
        return state_machine_codes.get(code, "unknown")

    def state_callback(msg: SportModeState_):  # type: ignore
        """
        Callback for receiving sport mode state messages.

        Parameters
        ----------
        msg: SportModeState_
        """
        go2_sport_mode_state_msg = msg  # type: ignore
        go2_state_code = msg.error_code  # type: ignore
        go2_state = get_state_from_code(msg.error_code)  # type: ignore
        go2_action_progress = msg.progress  # type: ignore

        data = {
            "go2_sport_mode_state_msg": go2_sport_mode_state_msg,
            "go2_state_code": go2_state_code,
            "go2_state": go2_state,
            "go2_action_progress": go2_action_progress,
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
        ChannelFactoryInitialize(0, channel)  # type: ignore
    except Exception as e:
        logging.error(f"Error initializing Unitree Go2 odom channel: {e}")
        return

    try:
        subscriber = ChannelSubscriber(channel, SportModeState_)  # type: ignore
        subscriber.Init(state_callback, 10)
        logging.info(f"Subscribed to {channel} for Unitree Go2 state data")
    except Exception as e:
        logging.error(f"Error subscribing to Unitree Go2 state channel: {e}")
        return

    running = True

    while running:
        try:
            cmd = control_queue.get_nowait()
            if cmd == "STOP":
                running = False
                break
        except Empty:
            pass

        time.sleep(0.1)

    logging.info("Unitree Go2 state processor stopped.")
    subscriber.Close()


@singleton
class UnitreeGo2StateProvider:
    """
    Unitree Go2 State Provider.

    Parameters
    ----------
    channel : str
        CycloneDDS channel to subscribe to for Unitree Go2 state data.
    """

    def __init__(self, channel: str = ""):
        """
        Initialize the Unitree Go2 State Provider.

        Parameters
        ----------
        channel : str, optional
            CycloneDDS channel to subscribe to for Unitree Go2 state data. Defaults to "".
        """
        self.channel = channel

        self.data_queue = mp.Queue(maxsize=5)
        self.control_queue = mp.Queue()

        self._go2_state_reader_thread = None
        self._go2_state_processor_thread = None
        self._stop_event = threading.Event()

        self.go2_sport_mode_state_msg = None
        self.go2_state = None
        self.go2_state_code = None
        self.go2_action_progress = 0

    def start(self):
        """
        Start the Unitree Go2 state provider.
        """
        if (
            not self._go2_state_processor_thread
            or not self._go2_state_processor_thread.is_alive()
        ):
            self._go2_state_reader_thread = mp.Process(
                target=go2_state_processor,
                args=(
                    self.channel,
                    self.data_queue,
                    self.control_queue,
                    get_logging_config(),
                ),
            )
            self._go2_state_reader_thread.start()
            logging.info("Unitree Go2 state reader started.")

        if (
            not self._go2_state_processor_thread
            or not self._go2_state_processor_thread.is_alive()
        ):
            self._go2_state_processor_thread = threading.Thread(
                target=self._go2_state_processor,
                daemon=True,
            )
            self._go2_state_processor_thread.start()
            logging.info("Unitree Go2 state processor started.")

    def stop(self):
        """
        Stop the Unitree Go2 state provider.
        """
        self._stop_event.set()

        if self._go2_state_reader_thread:
            self.control_queue.put("STOP")
            self._go2_state_reader_thread.join()
            logging.info("Unitree Go2 state reader stopped.")

        if self._go2_state_processor_thread:
            self._go2_state_processor_thread.join()
            logging.info("Unitree Go2 state processor stopped.")

    def _go2_state_processor(self):
        """
        Process the Unitree Go2 state data from the data queue.
        """
        while not self._stop_event.is_set():
            try:
                data = self.data_queue.get_nowait()

                self.go2_sport_mode_state_msg = data.get("go2_sport_mode_state_msg")
                self.go2_state = data.get("go2_state")
                self.go2_state_code = data.get("go2_state_code")
                self.go2_action_progress = data.get("go2_action_progress")

            except Empty:
                time.sleep(0.1)
                continue

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
