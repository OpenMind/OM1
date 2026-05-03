"""Unitree Go2 battery input plugin (Zenoh transport).

Reads ``LowState`` from a Zenoh keyexpression (default ``lowstate``) via
``open_zenoh_session()``. Surface-compatible with ``UnitreeGo2Battery``.

    agent_inputs: [
        { type: "UnitreeGo2BatteryZenoh" },
    ]

In simulation, only the battery-relevant fields of ``LowState`` are
populated; all others may be zero.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, IOProvider, TeleopsStatus, TeleopsStatusProvider
from zenoh_msgs import open_zenoh_session
from zenoh_msgs.idl.unitree_go import LowState


class UnitreeGo2BatteryZenohConfig(SensorConfig):
    """Configuration for ``UnitreeGo2BatteryZenoh``.

    Parameters
    ----------
    api_key : Optional[str]
        Teleops API key (forwarded to ``TeleopsStatusProvider``).
    topic : str
        Zenoh key for the LowState publication. Default ``lowstate``.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    topic: str = Field(
        default="lowstate",
        description="Zenoh key for Go2 LowState (unitree_go/msg/LowState).",
    )


class UnitreeGo2BatteryZenoh(FuserInput[UnitreeGo2BatteryZenohConfig, List[float]]):
    """Zenoh-routed Go2 battery monitor."""

    def __init__(self, config: UnitreeGo2BatteryZenohConfig):
        super().__init__(config)

        self.io_provider = IOProvider()
        self.status_provider = TeleopsStatusProvider(api_key=self.config.api_key)
        self.messages: list[Message] = []

        # Mirrors fields the legacy provider exposes.
        self.battery_percentage: float = 0.0
        self.battery_voltage: float = 0.0
        self.battery_amperes: float = 0.0
        self.battery_t: int = 0
        self._lock = threading.Lock()

        try:
            self._session = open_zenoh_session()
            self._session.declare_subscriber(self.config.topic, self._on_lowstate)
            logging.info("UnitreeGo2BatteryZenoh subscribed to '%s'", self.config.topic)
        except Exception:
            logging.exception("UnitreeGo2BatteryZenoh: failed to open Zenoh session")
            self._session = None

        self.descriptor_for_LLM = "Energy Levels"

    def _on_lowstate(self, sample) -> None:  # type: ignore[no-untyped-def]
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
        """Push the latest battery snapshot to the teleops status channel."""
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
        await asyncio.sleep(2.0)
        await self.report_status()
        with self._lock:
            pct, volt, amp = self.battery_percentage, self.battery_voltage, self.battery_amperes
        logging.info(f"Battery percentage: {pct} voltage: {volt} amperes: {amp}")
        return [pct, volt, amp]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
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
        """Append a formatted battery message to the input buffer."""
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return and clear the most recent formatted message."""
        if not self.messages:
            return None
        latest = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{latest.message}\n// END\n"
        self.io_provider.add_input(self.__class__.__name__, latest.message, latest.timestamp)
        self.messages = []
        return result
