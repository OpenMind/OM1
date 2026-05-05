import asyncio
import logging
import threading
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, IOProvider, TeleopsStatus, TeleopsStatusProvider
from zenoh_msgs import ZenohSampleType, open_zenoh_session
from zenoh_msgs.idl.unitree_go import LowState


class UnitreeGo2BatteryZenohConfig(SensorConfig):
    """
    Configuration for the Unitree Go2 Battery Zenoh Provider.

    Parameters
    ----------
    api_key : Optional[str]
        Teleops API key (forwarded to ``TeleopsStatusProvider``).
    topic : str
        Zenoh key for the LowState publication. Default ``lowstate``.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    topic: str = Field(
        default="lowstate",
        description="Zenoh key for Go2 LowState (unitree_go/msg/LowState).",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to use the simulation Zenoh endpoint instead of a local one.",
    )


class UnitreeGo2BatteryZenoh(FuserInput[UnitreeGo2BatteryZenohConfig, List[float]]):
    """Unitree Go2 Battery Zenoh Provider."""

    def __init__(self, config: UnitreeGo2BatteryZenohConfig):
        """
        Initialize the provider, set up the Zenoh subscriber for battery data, and prepare for status reporting.

        Parameters
        ----------
        config : UnitreeGo2BatteryZenohConfig
            Configuration for the provider.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.status_provider = TeleopsStatusProvider(api_key=self.config.api_key)
        self.messages: list[Message] = []

        self.battery_percentage: float = 0.0
        self.battery_voltage: float = 0.0
        self.battery_amperes: float = 0.0
        self.battery_t: int = 0
        self._lock = threading.Lock()

        try:
            self._session = open_zenoh_session()
            self._session.declare_subscriber(self.config.topic, self.LowStateMessageHandler)
            logging.info("UnitreeGo2BatteryZenoh subscribed to '%s'", self.config.topic)
        except Exception:
            logging.exception("UnitreeGo2BatteryZenoh: failed to open Zenoh session")
            self._session = None

        self.descriptor_for_LLM = "Energy Levels"

    def LowStateMessageHandler(self, sample: ZenohSampleType) -> None:
        """
        Handle incoming LowState messages from the Zenoh subscription, updating the internal battery status.

        Parameters
        ----------
        sample : ZenohSampleType
            The incoming Zenoh sample containing the LowState message payload.
        """
        try:
            msg = LowState.deserialize(sample.payload.to_bytes())
        except Exception:
            logging.exception("UnitreeGo2BatteryZenoh: decode failed")
            return
        try:
            with self._lock:
                self.battery_percentage = round(float(msg.bms_state.soc), 2)
                self.battery_voltage = round(float(msg.power_v), 2)
                self.battery_amperes = round(float(msg.power_a), 2)
                self.battery_t = int((msg.temperature_ntc1 + msg.temperature_ntc2) / 2)
        except AttributeError:
            logging.warning("UnitreeGo2BatteryZenoh: incomplete LowState message")

    async def report_status(self) -> None:
        """
        Push the latest battery snapshot to the teleops status channel.
        """
        with self._lock:
            level, t, v = self.battery_percentage, self.battery_t, self.battery_voltage
        self.status_provider.share_status(
            TeleopsStatus(
                machine_name="UnitreeGo2",
                update_time=str(time.time()),
                battery_status=BatteryStatus(
                    battery_level=level,
                    temperature=t,
                    voltage=v,
                    timestamp=str(time.time()),
                    charging_status=False,
                ),
            )
        )

    async def _poll(self) -> List[float]:
        """
        Poll for new battery data, report status, and return the latest battery snapshot as a list of floats.

        Returns
        -------
        List[float]
            A list containing battery percentage, voltage, and amperes.
        """
        await asyncio.sleep(2.0)
        await self.report_status()

        with self._lock:
            battery_percentage, battery_voltage, battery_amperes = (
                self.battery_percentage,
                self.battery_voltage,
                self.battery_amperes,
            )

        logging.info(f"Battery percentage: {battery_percentage} voltage: {battery_voltage} amperes: {battery_amperes}")

        return [battery_percentage, battery_voltage, battery_amperes]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """
        Convert the latest battery snapshot into a human-readable message with warnings if battery is low.

        Parameters
        ----------
        raw_input : List[float]
            A list containing battery percentage, voltage, and amperes.

        Returns
        -------
        Optional[Message]
            A Message object containing a human-readable description of the battery status, or None if battery is
        """
        battery_percentage = raw_input[0]

        if battery_percentage < 7:
            message = (
                "CRITICAL: Your battery is almost empty. Immediately move to your "
                "charging station and recharge. If you cannot find your charging "
                "station, consider sitting down."
            )
            return Message(timestamp=time.time(), message=message)

        if battery_percentage < 15:
            message = "WARNING: You are low on energy. Move to your charging station and " "recharge."
            return Message(timestamp=time.time(), message=message)

        return None

    async def raw_to_text(self, raw_input: List[float]) -> None:
        """
        Convert raw battery data into text and update the message buffer with any relevant warnings.

        Parameters
        ----------
        raw_input : List[float]
            A list containing battery percentage, voltage, and amperes.
        """
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format the latest message in the buffer for the LLM and log it to the IOProvider.

        Returns
        -------
        Optional[str]
            A formatted string containing the latest message for the LLM, or None if no messages are available.
        """
        if not self.messages:
            return None

        latest = self.messages[-1]

        result = f"""
{self.descriptor_for_LLM}: "{latest.message}"
"""
        self.io_provider.add_input(self.__class__.__name__, latest.message, latest.timestamp)

        self.messages = []
        return result
