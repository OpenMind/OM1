import asyncio
import logging
import time
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider


class UnitreeGo2LocationsSensorConfig(SensorConfig):
    """
    Configuration for Unitree Go2 Locations Sensor.

    Parameters
    ----------
    base_url : str
        Base URL for the locations service.
    timeout : int
        Timeout in seconds.
    refresh_interval : int
        Refresh interval in seconds.
    """

    base_url: str = Field(
        default="http://localhost:5000/maps/locations/list",
        description="Base URL for the locations service",
    )
    timeout: int = Field(default=5, description="Timeout in seconds")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds")


class UnitreeGo2LocationsInput(
    FuserInput[UnitreeGo2LocationsSensorConfig, Optional[str]]
):
    """
    Input plugin that publishes available saved locations for LLM prompts (Unitree Go2).

    Reads locations from IOProvider (populated by Locations background task).
    """

    def __init__(self, config: UnitreeGo2LocationsSensorConfig):
        """
        Initialize the UnitreeGo2LocationsInput plugin.

        Parameters
        ----------
        config : UnitreeGo2LocationsSensorConfig
            Configuration for the sensor input.
        """
        super().__init__(config)

        base_url = self.config.base_url
        timeout = self.config.timeout
        refresh_interval = self.config.refresh_interval

        self.locations_provider = UnitreeGo2LocationsProvider(
            base_url, timeout, refresh_interval
        )
        self.io_provider = IOProvider()

        self.messages: List[Message] = []
        self.descriptor_for_LLM = "These are the saved locations you can navigate to."

    async def _poll(self) -> Optional[str]:
        """
        Poll the UnitreeGo2LocationsProvider for the latest locations.

        Returns
        -------
        Optional[str]
            Formatted string of locations or None if no locations are available.
        """
        await asyncio.sleep(0.5)

        locations = self.locations_provider.get_all_locations()

        lines = []
        for name, entry in locations.items():
            label = entry.get("name") if isinstance(entry, dict) else name
            pose = entry.get("pose") if isinstance(entry, dict) else None
            if pose and isinstance(pose, dict):
                pos = pose.get("position", {})
                lines.append(f"{label} (x:{pos.get('x',0):.2f} y:{pos.get('y',0):.2f})")
            else:
                lines.append(f"{label}")

        result = "\n".join(lines)
        logging.debug(f"UnitreeGo2LocationsInput: formatted {len(lines)} locations")
        return result

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw input string to Message dataclass.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to processed text and manage buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed
        """
        if raw_input is None:
            return
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1].message}
// END
"""
        self.io_provider.add_input(
            self.__class__.__name__,
            self.messages[-1].message,
            self.messages[-1].timestamp,
        )

        # Reset messages buffer
        self.messages = []
        return result
