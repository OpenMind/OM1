"""
Status Heartbeat Plugin - Reports online status to Portal without requiring real hardware.
"""

import asyncio
import logging
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, TeleopsStatus, TeleopsStatusProvider


class StatusHeartbeatConfig(SensorConfig):
    """
    Configuration for Status Heartbeat.

    Parameters
    ----------
    api_key : Optional[str]
        API Key for OpenMind Portal.
    machine_name : str
        Name to display on Portal.
    """

    api_key: Optional[str] = Field(default=None, description="API Key")
    machine_name: str = Field(default="OM1 Agent", description="Machine name on Portal")


class StatusHeartbeat(FuserInput[StatusHeartbeatConfig, List[float]]):
    """
    Status Heartbeat - Reports online status to Portal.

    This plugin periodically sends a heartbeat to the OpenMind Portal
    to show the machine as "Online" without requiring real robot hardware.
    """

    def __init__(self, config: StatusHeartbeatConfig):
        """
        Initialize Status Heartbeat.
        """
        super().__init__(config)

        api_key = self.config.api_key
        self.machine_name = self.config.machine_name

        # Status provider
        self.status_provider = TeleopsStatusProvider(api_key=api_key)

        # Messages buffer
        self.messages: list[Message] = []

        # Simulated battery state (100% since it's a computer)
        self.battery_percentage = 100.0

        # Simple description
        self.descriptor_for_LLM = "System Status"

        logging.info(f"Status Heartbeat initialized for: {self.machine_name}")

    async def report_status(self):
        """
        Report the status to the Portal.
        """
        try:
            self.status_provider.share_status(
                TeleopsStatus(
                    machine_name=self.machine_name,
                    update_time=str(time.time()),
                    battery_status=BatteryStatus(
                        battery_level=self.battery_percentage,
                        temperature=25.0,  # Room temperature
                        voltage=12.0,
                        timestamp=str(time.time()),
                        charging_status=True,  # Always "charging" since it's a computer
                    ),
                )
            )
            logging.debug(f"Status heartbeat sent for {self.machine_name}")
        except Exception as e:
            logging.error(f"Error sending status heartbeat: {e}")

    async def _poll(self) -> List[float]:
        """
        Poll and send heartbeat.

        Returns
        -------
        List[float]
            Simulated status data
        """
        await asyncio.sleep(5.0)  # Send heartbeat every 5 seconds
        await self.report_status()

        return [self.battery_percentage]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """
        Process raw status (not used for LLM input).
        """
        return None

    async def raw_to_text(self, raw_input: List[float]):
        """
        Convert raw status to text (not used).
        """
        pass

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return None - this plugin doesn't provide LLM input.
        """
        return None
