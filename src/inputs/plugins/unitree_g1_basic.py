import asyncio
import logging
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, IOProvider, TeleopsStatus, TeleopsStatusProvider

try:
    from unitree.unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
    from unitree.unitree_sdk2py.idl.unitree_hg.msg import dds_  # type: ignore
except ImportError:
    logging.warning(
        "Unitree SDK not found. Please install the Unitree SDK to use this plugin."
    )

    class BmsState_:
        """
        Placeholder for BmsState_ when Unitree SDK is not installed.
        """

        def __init__(self):
            pass

    class LowState_:
        """
        Placeholder for LowState_ when Unitree SDK is not installed.
        """

        def __init__(self):
            pass


# Data structure documentation:
#
# class BmsState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.BmsState_"):
#     version_high: types.uint8
#     version_low: types.uint8
#     fn: types.uint8
#     cell_vol: types.array[types.uint16, 40]
#     bmsvoltage: types.array[types.uint32, 3]
#     current: types.int32
#     soc: types.uint8
#     soh: types.uint8
#     temperature: types.array[types.int16, 12]
#     cycle: types.uint16
#     manufacturer_date: types.uint16
#     bmsstate: types.array[types.uint32, 5]
#     reserve: types.array[types.uint32, 3]
#
# class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
#     version: types.array[types.uint32, 2]
#     mode_pr: types.uint8
#     mode_machine: types.uint8
#     tick: types.uint32
#     imu_state: 'unitree.unitree_sdk2py.idl.unitree_hg.msg.dds_.IMUState_'
#     motor_state: types.array['unitree.unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorState_', 35]
#     wireless_remote: types.array[types.uint8, 40]


class UnitreeG1BasicConfig(SensorConfig):
    """
    Configuration for Unitree G1 Basic Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    unitree_ethernet : Optional[str]
        Unitree Ethernet Interface.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet Interface"
    )


class UnitreeG1Basic(FuserInput[UnitreeG1BasicConfig, List[float]]):
    """
    Unitree G1 Basic Functionality.

    Takes specific Unitree CycloneDDS Lowstate messages, converts them to
    text strings, and sends them to the fuser.

    Processes Unitree Lowstate information.
    These are things like joint position and battery charge.

    Maintains a buffer of processed messages.
    """

    def __init__(self, config: UnitreeG1BasicConfig):
        """
        Initialize Unitree bridge with empty message buffer.
        """
        super().__init__(config)

        api_key = self.config.api_key

        # IO provider
        self.io_provider = IOProvider()

        # Status provider
        self.status_provider = TeleopsStatusProvider(api_key=api_key)

        # Messages buffer
        self.messages: list[Message] = []

        # create subscriber
        self.low_state = None
        self.lowstate_subscriber = None
        self.bmsstate_subscriber = None

        unitree_ethernet = self.config.unitree_ethernet
        logging.info(f"UnitreeG1Basic using ethernet: {unitree_ethernet}")

        # Joint angles e.g.
        if unitree_ethernet and unitree_ethernet != "":
            # only set up if we are connected to a robot
            self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # type: ignore
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)

            # Battery specific data
            self.bmsstate_subscriber = ChannelSubscriber("rt/lf/bmsstate", BmsState_)  # type: ignore
            self.bmsstate_subscriber.Init(self.BMSStateHandler, 10)

        # battery state
        self.battery_percentage = 0.0
        self.battery_voltage = 0.0
        self.battery_amperes = 0.0
        self.battery_temperature = 0

        self.g1_lowbatt_percent = 20.0  # percent
        self.descriptor_for_LLM = "Energy Level"

    def BMSStateHandler(self, msg: dds_.BmsState_):  # type: ignore
        """
        Handle incoming BmsState messages from the Unitree robot.

        Parameters
        ----------
        msg : dds_.BmsState_
            The incoming BmsState message containing battery state information.
        """
        self.bms_state = msg
        logging.debug(f"BmsState_: {msg}")

        self.battery_voltage = float(msg.bmsvoltage[0])  # type: ignore
        self.battery_amperes = float(msg.current)  # type: ignore
        self.battery_percentage = float(msg.soc)  # type: ignore
        self.battery_temperature = float(msg.temperature[0])  # type: ignore

    def LowStateHandler(self, msg: dds_.LowState_):  # type: ignore
        """
        Handle incoming LowState messages from the Unitree robot.

        Parameters
        ----------
        msg : dds_.LowState_
            The incoming LowState message containing robot state information.
        """
        self.low_state = msg
        logging.debug(f"LowState_: {msg}")

    async def update_status(self):
        """
        Report the battery status to the status provider.
        """
        self.status_provider.share_status(
            TeleopsStatus(
                machine_name="UnitreeG1",
                update_time=str(time.time()),
                battery_status=BatteryStatus(
                    battery_level=self.battery_percentage,
                    temperature=self.battery_temperature,
                    voltage=self.battery_voltage,
                    timestamp=str(time.time()),
                    charging_status=False,
                ),
            )
        )

    async def _poll(self) -> List[float]:
        """
        Poll for new lowstate data.

        Returns
        -------
        List[float]
            list of floats
        """
        await asyncio.sleep(2.0)

        await self.update_status()

        # logging.info(f"Battery voltage: {self.latest_v} current: {self.latest_a}")

        return [self.battery_percentage, self.battery_voltage, self.battery_amperes]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """
        Process raw lowstate to generate text description.

        Parameters
        ----------
        raw_input : List[float]
            Raw lowstate data to be processed

        Returns
        -------
        Message
            Timestamped message containing description
        """
        battery_percentage = raw_input[0]

        if battery_percentage < self.g1_lowbatt_percent:
            message = "WARNING: You are low on energy. SIT DOWN NOW."
            return Message(timestamp=time.time(), message=message)
        elif (
            battery_percentage < self.g1_lowbatt_percent + 10.0
        ):  # the +10% is an emergency reserve
            message = "WARNING: You are low on energy. Consider sitting down."
            return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: List[float]):
        """
        Convert raw lowstate to text and update message buffer.

        Parameters
        ----------
        raw_input : List[float]
            Raw lowstate data to be processed
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with timestamp and class name,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
