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
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except ImportError:
    logging.warning(
        "Unitree SDK or CycloneDDS not found. You do not need this unless you are connecting to a Unitree robot."
    )

from .thermal_provider_base import ThermalProviderBase, RobotState
from .singleton import singleton


def g1_thermal_processor(
    channel: str,
    data_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Process function for the Unitree G1 Thermal Provider.
    This function runs in a separate process to periodically retrieve the thermal
    data from the robot via CycloneDDS and put it into a multiprocessing queue.

    Parameters
    ----------
    channel : str
        The channel to connect to the robot.
    data_queue : mp.Queue
        Queue for sending the retrieved thermal data.
    logging_config : LoggingConfig, optional
        Optional logging configuration. If provided, it will override the default logging settings.
    """
    setup_logging("g1_thermal_processor", logging_config=logging_config)

    def low_state_handler(data: LowState_):  # type: ignore
        """
        Handler for LowState messages from CycloneDDS.

        Parameters
        ----------
        data : LowState_
            The LowState message containing thermal data.
        """
        logging.debug(f"LowState handler: {data}")  # type: ignore
        data_queue.put(data)  # type: ignore

    try:
        ChannelFactoryInitialize(0, channel)  # type: ignore
    except Exception as e:
        logging.error(f"Error initializing Unitree G1 thermal channel: {e}")
        return

    try:
        low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # type: ignore
        low_state_subscriber.Init(low_state_handler, 10)
        logging.info("CycloneDDS LowState subscriber initialized successfully")
    except Exception as e:
        logging.error(f"Error opening CycloneDDS client: {e}")
        return None

    while True:
        time.sleep(0.1)


@singleton
class UnitreeG1ThermalProvider(ThermalProviderBase):
    """
    Unitree G1 Thermal Provider.

    This class implements thermal management for Unitree G1 robots using CycloneDDS
    for communication.

    Parameters
    ----------
    channel : str
        The channel to connect to the robot, used for CycloneDDS.
    """

    def __init__(self, channel: Optional[str] = None):
        """
        Initialize the Unitree G1 Thermal Provider.

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
        Start the Unitree G1 Thermal Provider.
        """
        if self._thermal_reader_thread and self._thermal_reader_thread.is_alive():
            logging.warning("G1 Thermal Provider is already running.")
            return

        if not self.channel:
            logging.error("Channel must be specified to start the G1 Thermal Provider.")
            return

        logging.info(f"Starting Unitree G1 Thermal Provider on channel: {self.channel}")

        self._thermal_reader_thread = mp.Process(
            target=g1_thermal_processor,
            args=(
                self.channel,
                self.data_queue,
                get_logging_config(),
            ),
            daemon=True,
        )
        self._thermal_reader_thread.start()

        if self._thermal_processor_thread and self._thermal_processor_thread.is_alive():
            logging.warning("Thermal processor thread is already running.")
            return
        else:
            logging.info("Starting Thermal processor thread")
            self._thermal_processor_thread = threading.Thread(target=self.process_thermal, daemon=True)
            self._thermal_processor_thread.start()

    def process_thermal(self):
        """
        Process the G1 LowState data and update the internal state.
        This overrides the base class method to handle G1-specific message format.
        """
        import math

        while not self._stop_event.is_set():
            try:
                low_data = self.data_queue.get(timeout=1)
            except Exception:
                continue

            # Extract timestamp
            self.thermal_subscriber_ts = time.time()

            # motor_state: types.array['unitree.unitree_sdk2py.idl.unitree_go.msg.dds_.MotorState_', 20]
            # temperature_ntc1: types.uint8
            # temperature_ntc2: types.uint8

            # Update current temperature
            self.m0 = round(low_data.motor_state[0], 2)
            self.m1 = round(low_data.motor_state[1], 2)

            logging.debug(
                f"G1 Thermal: M0:{self.m0} M1:{self.m1}"
            )
