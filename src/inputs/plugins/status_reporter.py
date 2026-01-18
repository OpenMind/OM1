import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers import BatteryStatus, TeleopsStatus, TeleopsStatusProvider
from providers.io_provider import IOProvider


class StatusReporter(FuserInput[SensorConfig, Optional[str]]):
    """
    Reports robot status to OpenMind control panel.
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.io_provider = IOProvider()
        
        # Get API key from environment or config
        import os
        api_key = os.getenv("OM_API_KEY")
        
        if not api_key:
            logging.error("OM_API_KEY environment variable is required for StatusReporter")
            return
        
        # Initialize status provider
        self.status_provider = TeleopsStatusProvider(api_key=api_key)
        
        self.descriptor_for_LLM = "StatusReporter"
        logging.info("StatusReporter initialized - will report to OpenMind control panel")

    async def _poll(self) -> Optional[str]:
        """
        Poll and report status every 30 seconds.
        """
        await asyncio.sleep(30)  # Report every 30 seconds
        
        # Create status report
        battery_status = BatteryStatus(
            battery_level=85.0,  # Mock battery level
            temperature=25.0,
            voltage=12.0,
            timestamp=str(time.time()),
            charging_status=False
        )
        
        status = TeleopsStatus(
            update_time=str(time.time()),
            battery_status=battery_status,
            machine_name="OM1_Conversation_Bot",
            video_connected=True  # We have camera working
        )
        
        # Report status
        self.status_provider.share_status(status)
        logging.info("Status reported to OpenMind control panel")
        
        return "status_reported"

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process status report.
        """
        if raw_input is None:
            return None
        
        return Message(timestamp=time.time(), message="Robot status reported to control panel")

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.
        """
        # We don't need to store these messages
        pass

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        No need to format status reports for LLM.
        """
        return None
