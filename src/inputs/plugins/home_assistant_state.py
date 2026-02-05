"""
Home Assistant Device State Input Plugin for OM1.

This module provides an input plugin that monitors Home Assistant device states
and reports them to the OM1 fuser for processing by the LLM.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class HomeAssistantInputConfig(SensorConfig):
    """
    Configuration for Home Assistant Input sensor.

    Parameters
    ----------
    base_url : str
        Home Assistant URL (e.g., 'http://homeassistant.local:8123').
    access_token : str
        Long-lived access token from Home Assistant.
    entity_ids : List[str]
        List of entity IDs to monitor (e.g., ['light.living_room', 'climate.bedroom']).
    poll_interval : float
        Polling interval in seconds.
    verify_ssl : bool
        Whether to verify SSL certificates.
    report_all_states : bool
        If True, report all device states. If False, only report changes.
    input_name : str
        Name of the input for LLM prompt.
    """

    base_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant base URL",
    )
    access_token: str = Field(
        default="",
        description="Long-lived access token from Home Assistant",
    )
    entity_ids: List[str] = Field(
        default_factory=list,
        description="List of entity IDs to monitor",
    )
    poll_interval: float = Field(
        default=5.0,
        description="Polling interval in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )
    report_all_states: bool = Field(
        default=False,
        description="Report all states or only changes",
    )
    input_name: str = Field(
        default="Smart Home Devices",
        description="Name of the input for LLM prompt",
    )


@dataclass
class DeviceState:
    """
    Container for device state information.

    Parameters
    ----------
    entity_id : str
        Home Assistant entity ID.
    state : str
        Current state value.
    friendly_name : str
        Human-friendly device name.
    attributes : Dict[str, Any]
        Additional state attributes.
    last_changed : str
        Timestamp of last state change.
    """

    entity_id: str
    state: str
    friendly_name: str
    attributes: Dict[str, Any]
    last_changed: str


class HomeAssistantStateInput(FuserInput[HomeAssistantInputConfig, Optional[str]]):
    """
    Input plugin that monitors Home Assistant device states.

    This plugin polls Home Assistant for the current state of configured
    devices and reports them to the fuser for processing by the LLM.
    The robot can then react to changes in the smart home environment.
    """

    def __init__(self, config: HomeAssistantInputConfig):
        """
        Initialize the Home Assistant state input.

        Parameters
        ----------
        config : HomeAssistantInputConfig
            Configuration for the input.
        """
        super().__init__(config)

        self.base_url = config.base_url.rstrip("/")
        self.access_token = config.access_token
        self.entity_ids = config.entity_ids
        self.poll_interval = config.poll_interval
        self.verify_ssl = config.verify_ssl
        self.report_all_states = config.report_all_states
        self.descriptor_for_LLM = config.input_name

        self.io_provider = IOProvider()
        self._session: Optional[aiohttp.ClientSession] = None
        self._previous_states: Dict[str, str] = {}
        self._last_poll_time: float = 0
        self.messages: List[Message] = []

        if not self.access_token:
            logging.warning(
                "HomeAssistant state input: access_token is empty. "
                "API requests will fail. Set a long-lived access token "
                "from your Home Assistant instance."
            )

        logging.info(
            f"HomeAssistant state input initialized for {self.base_url} "
            f"monitoring {len(self.entity_ids)} entities"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Home Assistant API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_headers(),
            )
        return self._session

    async def _get_entity_state(self, entity_id: str) -> Optional[DeviceState]:
        """
        Get the current state of an entity.

        Parameters
        ----------
        entity_id : str
            Home Assistant entity ID.

        Returns
        -------
        Optional[DeviceState]
            Device state or None if not found.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/states/{entity_id}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return DeviceState(
                        entity_id=entity_id,
                        state=data.get("state", "unknown"),
                        friendly_name=data.get("attributes", {}).get(
                            "friendly_name", entity_id
                        ),
                        attributes=data.get("attributes", {}),
                        last_changed=data.get("last_changed", ""),
                    )
                else:
                    logging.warning(
                        f"Could not get state for {entity_id}: HTTP {response.status}"
                    )
                    return None
        except Exception as e:
            logging.error(f"Error getting entity state for {entity_id}: {e}")
            return None

    async def _get_all_states(self) -> List[DeviceState]:
        """
        Get states for all monitored entities.

        Returns
        -------
        List[DeviceState]
            List of device states.
        """
        states = []
        for entity_id in self.entity_ids:
            state = await self._get_entity_state(entity_id)
            if state:
                states.append(state)
        return states

    def _format_state_for_llm(self, state: DeviceState) -> str:
        """
        Format a device state for LLM consumption.

        Parameters
        ----------
        state : DeviceState
            Device state to format.

        Returns
        -------
        str
            Human-readable state description.
        """
        entity_type = state.entity_id.split(".")[0]
        name = state.friendly_name

        # Format based on entity type
        if entity_type == "light":
            if state.state == "on":
                brightness = state.attributes.get("brightness", 255)
                brightness_pct = round(brightness / 255 * 100)
                color = state.attributes.get("rgb_color")
                if color:
                    return f"{name} is ON at {brightness_pct}% brightness (color: RGB {color})"
                return f"{name} is ON at {brightness_pct}% brightness"
            return f"{name} is OFF"

        elif entity_type == "switch":
            return f"{name} is {'ON' if state.state == 'on' else 'OFF'}"

        elif entity_type == "climate":
            temp = state.attributes.get("temperature")
            current_temp = state.attributes.get("current_temperature")
            mode = state.state
            if current_temp and temp:
                return (
                    f"{name} is in {mode} mode, "
                    f"current: {current_temp}°, target: {temp}°"
                )
            elif mode:
                return f"{name} is in {mode} mode"
            return f"{name} state: {state.state}"

        elif entity_type == "sensor":
            unit = state.attributes.get("unit_of_measurement", "")
            return f"{name}: {state.state} {unit}".strip()

        elif entity_type == "binary_sensor":
            return f"{name} is {'detected' if state.state == 'on' else 'not detected'}"

        elif entity_type == "cover":
            position = state.attributes.get("current_position")
            if position is not None:
                return f"{name} is {state.state} ({position}% open)"
            return f"{name} is {state.state}"

        elif entity_type == "fan":
            if state.state == "on":
                speed = state.attributes.get("percentage", 100)
                return f"{name} is ON at {speed}% speed"
            return f"{name} is OFF"

        else:
            return f"{name}: {state.state}"

    def _format_all_states(self, states: List[DeviceState]) -> str:
        """
        Format all device states into a single message.

        Parameters
        ----------
        states : List[DeviceState]
            List of device states.

        Returns
        -------
        str
            Formatted message describing all device states.
        """
        if not states:
            return "No smart home devices available."

        lines = ["Current smart home device status:"]
        for state in states:
            lines.append(f"  - {self._format_state_for_llm(state)}")

        return "\n".join(lines)

    def _get_changed_states(
        self, current_states: List[DeviceState]
    ) -> List[DeviceState]:
        """
        Filter states to only include those that have changed.

        Note: This compares only the primary state string (e.g., 'on'/'off'),
        not device attributes (brightness, temperature, etc.). Attribute-only
        changes (e.g., brightness change while state remains 'on') will not
        be detected as changes.

        Parameters
        ----------
        current_states : List[DeviceState]
            Current device states.

        Returns
        -------
        List[DeviceState]
            States that have changed since last poll.
        """
        changed = []
        for state in current_states:
            prev_state = self._previous_states.get(state.entity_id)
            if prev_state != state.state:
                changed.append(state)
                self._previous_states[state.entity_id] = state.state
        return changed

    async def _poll(self) -> Optional[str]:
        """
        Poll Home Assistant for device states.

        Returns
        -------
        Optional[str]
            Formatted state message or None if no updates.
        """
        current_time = time.time()

        # Respect poll interval
        if current_time - self._last_poll_time < self.poll_interval:
            remaining = self.poll_interval - (current_time - self._last_poll_time)
            await asyncio.sleep(remaining)
            return None

        self._last_poll_time = current_time

        try:
            states = await self._get_all_states()

            if not states:
                return None

            if self.report_all_states:
                return self._format_all_states(states)
            else:
                changed_states = self._get_changed_states(states)
                if changed_states:
                    lines = ["Smart home device changes detected:"]
                    for state in changed_states:
                        lines.append(f"  - {self._format_state_for_llm(state)}")
                    return "\n".join(lines)

            return None

        except Exception as e:
            logging.error(f"Error polling Home Assistant: {e}")
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw input to a Message object.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw state information.

        Returns
        -------
        Optional[Message]
            Message object or None.
        """
        if raw_input is None:
            return None

        return Message(
            timestamp=time.time(),
            message=raw_input,
        )

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw state information.
        """
        message = await self._raw_to_text(raw_input)
        if message is not None:
            self.messages.append(message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with the standard OM1 input format,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty.
        """
        if not self.messages:
            return None

        latest = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest.message, latest.timestamp
        )
        self.messages = []

        return result

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
