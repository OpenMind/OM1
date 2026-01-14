import asyncio
import logging
import time
from typing import List, Optional

import zenoh
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, IOProvider, TeleopsStatus, TeleopsStatusProvider
from zenoh_msgs import open_zenoh_session, sensor_msgs


class TurtleBot4BatteryConfig(SensorConfig):
    """
    Configuration for TurtleBot4 battery sensor.

    Parameters
    ----------
    api_key : Optional[str]
        API Key.
    URID : str
        URID.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    URID: str = Field(default="default", description="URID")


class TurtleBot4Battery(FuserInput[TurtleBot4BatteryConfig, List[str]]):
    """
    TurtleBot4 Battery inputs.

    Takes specific TurtleBot4 ROS2/Zenoh messages, converts them to
    text strings, and sends them to the fuser.

    Maintains a buffer of processed messages.
    """

    def __init__(self, config: TurtleBot4BatteryConfig):
        """
        Initialize the TurtleBot4Battery input handler.

        Sets up the required providers, buffers, and status variables
        for handling TurtleBot4 battery data.

        Parameters
        ----------
        config : TurtleBot4BatteryConfig
            Configuration for the sensor input.
        """
        super().__init__(config)

        api_key = self.config.api_key

        # IO provider
        self.io_provider = IOProvider()

        # Status provider
        self.status_provider = TeleopsStatusProvider(api_key=api_key)

        # Buffer for storing the final output
        self.messages: list[Message] = []

        # Status variables
        self.battery_status = None

        # Battery status variables
        self.battery_percentage = 0.0
        self.battery_voltage = 0.0
        self.battery_temperature = 0
        self.battery_timestamp = time.time()
        self.is_docked = False

        logging.info("Opening Zenoh TurtleBot4 session...")
        self.z = open_zenoh_session()

        logging.info(f"Config: {self.config}")

        self.URID = self.config.URID
        logging.info(f"Using TurtleBot4 URID: {self.URID}")

        logging.info("Creating Zenoh TurtleBot4 Subscribers")
        self.batts = self.z.declare_subscriber(
            f"{self.URID}/c3/battery_state", self.listener_battery
        )
        self.dock = self.z.declare_subscriber(
            f"{self.URID}/c3/dock_status", self.listener_dock
        )

        # Simple description of sensor output to help LLM understand its importance and utility
        self.descriptor_for_LLM = "Energy Level"

    def listener_battery(self, sample: zenoh.Sample):
        """
        Zenoh callback for battery state.

        Parameters
        ----------
        sample : zenoh.Sample
            Zenoh sample containing battery state data
        """
        battery = sensor_msgs.BatteryState.deserialize(sample.payload.to_bytes())

        logging.debug(f"TB4 battery callback: {battery}")

        self.battery_percentage = int(battery.percentage * 100)
        self.battery_voltage = battery.voltage
        self.battery_temperature = round(battery.temperature, 2)
        self.battery_timestamp = battery.header.stamp.sec

        if self.battery_percentage < 5:
            self.battery_status = (
                "CRITICAL: your battery is almost empty. "
                "Immediately move to your charging station and recharge."
            )
        elif self.battery_percentage < 15:
            self.battery_status = (
                "IMPORTANT: your battery is low. "
                "Consider finding your charging station and recharging."
            )
        else:
            self.battery_status = None

    def listener_dock(self, sample: zenoh.Sample):
        """
        Process docking status updates.

        Parameters
        ----------
        sample : zenoh.Sample
            Zenoh sample containing docking status data
        """
        dock = sensor_msgs.DockStatus.deserialize(sample.payload.to_bytes())
        self.is_docked = dock.is_docked

    async def report_status(self):
        """
        Report the battery status to the status provider.
        """
        self.status_provider.share_status(
            TeleopsStatus(
                machine_name="TurtleBot4",
                update_time=str(time.time()),
                battery_status=BatteryStatus(
                    battery_level=self.battery_percentage,
                    temperature=self.battery_temperature,
                    voltage=self.battery_voltage,
                    timestamp=str(self.battery_timestamp),
                    charging_status=self.is_docked,
                ),
            )
        )

    async def _poll(self) -> List[str]:
        """
        Poll for new state data.

        Returns
        -------
        List[str]
        """
        await asyncio.sleep(2.0)
        await self.report_status()

        logging.info(
            f"TB4 batt percent:{self.battery_percentage} low?: {self.battery_status}"
        )

        if self.battery_status is not None:
            return [self.battery_status]
        else:
            return []

    async def _raw_to_text(self, raw_input: List[str]) -> Optional[Message]:
        """
        Process raw lowstate to generate text description.

        Parameters
        ----------
        raw_input : List[str]
            State data to be processed

        Returns
        -------
        Optional[Message]
            Timestamped message containing description
        """
        if raw_input and raw_input[0]:
            return Message(timestamp=time.time(), message=raw_input[0])

        return None

    async def raw_to_text(self, raw_input: List[str]):
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
