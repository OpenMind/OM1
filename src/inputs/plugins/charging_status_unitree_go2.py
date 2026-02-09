import asyncio
import logging
import time
from typing import List, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.unitree_go2_charging_provider import UnitreeGo2ChargingProvider


class ChargingStatusUnitreeGo2(FuserInput[SensorConfig, str]):
    """
    Charging status input plugin for LLM prompts.

    Monitors the robot's charging status via Zenoh and provides
    clear feedback to the LLM about the current charging state.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize the ChargingStatusUnitreeGo2 plugin.

        Parameters
        ----------
        config : SensorConfig
            Configuration for the sensor input.
        """
        super().__init__(config)

        # Initialize providers
        self.charging_provider: UnitreeGo2ChargingProvider = (
            UnitreeGo2ChargingProvider()
        )
        self.charging_provider.start()
        self.io_provider = IOProvider()

        # Message buffer
        self.messages: List[Message] = []

        # Descriptive text for LLM context
        self.descriptor_for_LLM = (
            "Robot charging status - indicates the current battery charging state."
        )

        logging.info("ChargingStatusUnitreeGo2 plugin initialized")

    async def _poll(self) -> str:
        """
        Poll the Charging provider for charging status.

        Returns
        -------
        str
            Status message indicating the current charging state,
            or empty string if no status is available.
        """
        await asyncio.sleep(0.1)  # Brief delay to prevent excessive polling

        try:
            status = self.charging_provider.charging_status

            if status is None:
                return ""

            status_map = {
                0: "DISCHARGING: Robot is running on battery power.",
                1: "CHARGING: Robot is currently charging.",
                2: "ENROUTE_CHARGING: Robot is moving to charging station.",
                3: "FULLY_CHARGED: Robot battery is fully charged.",
            }

            status_msg = status_map.get(
                status, f"UNKNOWN: Unrecognized charging status ({status})."
            )
            logging.debug(f"Charging status: {status_msg}")

            return status_msg

        except Exception as e:
            logging.error(f"Error polling charging status: {e}")
            return "CHARGING ERROR: Unable to determine charging state."

    async def _raw_to_text(self, raw_input: str) -> Optional[Message]:
        """
        Convert raw input string to Message dataclass.

        Parameters
        ----------
        raw_input : str
            Raw charging status string

        Returns
        -------
        Optional[Message]
            Message dataclass containing the status and timestamp
        """
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent message from the buffer, formats it
        with timestamp and class name, adds it to the IO provider,
        and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest message and metadata,
            or None if the buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
