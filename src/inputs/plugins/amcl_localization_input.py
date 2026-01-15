import asyncio
import logging
import time
from typing import List, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider


class AMCLLocalizationInput(FuserInput[SensorConfig, Optional[str]]):
    """
    AMCL Localization status input plugin for LLM prompts.

    Monitors the robot's localization status via AMCL and provides
    clear feedback to the LLM about whether navigation is safe to proceed.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize the LocalizationInput plugin.

        Parameters
        ----------
        config : SensorConfig
            Configuration for the sensor input.
        """
        super().__init__(config)

        # Initialize providers
        self.amcl_provider: UnitreeGo2AMCLProvider = UnitreeGo2AMCLProvider()
        self.io_provider = IOProvider()

        # Message buffer
        self.messages: List[Message] = []

        # Descriptive text for LLM context
        self.descriptor_for_LLM = (
            "Robot localization status - indicates if navigation is safe to proceed."
        )

        logging.info("LocalizationInput plugin initialized")

    async def _poll(self) -> Optional[str]:
        """
        Poll the AMCL provider for localization status.

        Returns
        -------
        Optional[str]
            Status message indicating if robot is localized and ready for navigation,
            or None if no status change occurred.
        """
        await asyncio.sleep(0.1)  # Brief delay to prevent excessive polling

        try:
            is_localized = self.amcl_provider.is_localized
            pose = self.amcl_provider.pose

            if is_localized and pose is not None:
                status_msg = "LOCALIZED: Robot position is confirmed. Navigation commands are safe to execute."
                pos = pose.position
                logging.debug(
                    f"Robot localized at position x:{pos.x:.2f}, y:{pos.y:.2f}, z:{pos.z:.2f}"
                )
            else:
                status_msg = "NOT LOCALIZED: Robot position uncertain. DO NOT attempt navigation until localized."
                logging.debug("Robot localization status: NOT LOCALIZED")

            return status_msg

        except Exception as e:
            logging.error(f"Error polling localization status: {e}")
            return "LOCALIZATION ERROR: Unable to determine robot position. Navigation not recommended."

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw input string to Message dataclass.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw localization status string

        Returns
        -------
        Optional[Message]
            Message dataclass containing the status and timestamp
        """
        if raw_input is None:
            return None

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
