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
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_  # type: ignore
except ImportError:
    logging.warning(
        "Unitree SDK not found. Please install the Unitree SDK to use this plugin."
    )

    class ChannelSubscriber:
        """
        Placeholder for ChannelSubscriber when Unitree SDK is not installed.
        """

        def __init__(self):
            pass

    class LowState_:
        """
        Placeholder for LowState_ when Unitree SDK is not installed.
        """

        def __init__(self):
            pass


class UnitreeGo2BatteryConfig(SensorConfig):
    """
    Configuration for Unitree Go2 Battery Sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")


class UnitreeGo2Battery(FuserInput[UnitreeGo2BatteryConfig, List[float]]):
    """
    Unitree Go2 Lowstate bridge.

    Takes specific Unitree CycloneDDS Lowstate messages, converts them to
    text strings, and sends them to the fuser.

    Processes Unitree battery information.

    Maintains a buffer of processed messages.
    """

    def __init__(self, config: UnitreeGo2BatteryConfig):
        """
        Initialize Unitree bridge with empty message buffer.

        Parameters
        ----------
        config : UnitreeGo2BatteryConfig
            Configuration settings for the sensor input.
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

        try:
            self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # type: ignore
            logging.info("Battery monitor initialized")
        except Exception as e:
            logging.error(f"Error initializing Battery monitor: {e}")

        if self.lowstate_subscriber:
            self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)  # type: ignore

        # battery state
        self.battery_percentage = 0.0
        self.battery_voltage = 0.0
        self.battery_amperes = 0.0
        self.battery_t = 0

        # Simple description of sensor output to help LLM understand its importance and utility
        self.descriptor_for_LLM = "Energy Levels"

    def LowStateMessageHandler(self, msg: LowState_):
        """
        Handle incoming LowState messages from Unitree Go2.

        Parameters
        ----------
        msg : LowState_
            Incoming LowState message
        """
        try:
            self.low_state = msg
            self.battery_percentage = round(float(msg.bms_state.soc), 2)  # type: ignore
            self.battery_voltage = round(float(msg.power_v), 2)  # type: ignore
            self.battery_amperes = round(float(msg.power_a), 2)  # type: ignore
            self.battery_t = int((msg.temperature_ntc1 + msg.temperature_ntc2) / 2)  # type: ignore
        except AttributeError as e:
            logging.warning(f"Incomplete LowState message: {e}")
            self.battery_percentage = 0.0
            self.battery_voltage = 0.0
            self.battery_amperes = 0.0
            self.battery_t = 0

    async def report_status(self):
        """
        Report the battery status to the status provider.
        """
        self.status_provider.share_status(
            TeleopsStatus(
                machine_name="UnitreeGo2",
                update_time=str(time.time()),
                battery_status=BatteryStatus(
                    battery_level=self.battery_percentage,
                    temperature=self.battery_t,
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
        await self.report_status()

        logging.info(
            (
                f"Battery percentage: {self.battery_percentage} "
                f"voltage: {self.battery_voltage} "
                f"amperes: {self.battery_amperes}"
            )
        )

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
        Optional[Message]
            Timestamped message containing description
        """
        battery_percentage = raw_input[0]
        logging.debug(f"Battery percentage: {battery_percentage}")

        if battery_percentage < 7:
            message = "CRITICAL: Your battery is almost empty. Immediately move to your charging station and recharge. If you cannot find your charging station, consider sitting down."
            return Message(timestamp=time.time(), message=message)
        elif battery_percentage < 15:
            message = "WARNING: You are low on energy. Move to your charging station and recharge."
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

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
