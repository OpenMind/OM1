import asyncio
import logging
import time
from typing import Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class HomeAssistantInputConfig(SensorConfig):
    """
    Configuration for Home Assistant State Input.

    Parameters
    ----------
    base_url : str
        Home Assistant base URL (e.g. http://homeassistant.local:8123).
    token : str
        Long-lived access token from Home Assistant profile.
    entity_ids : str
        Comma-separated list of entity IDs to monitor
        (e.g. light.living_room,switch.fan,climate.bedroom).
    poll_interval : float
        Seconds between state updates (default: 30).
    """

    base_url: str = Field(default="", description="Home Assistant base URL")
    token: str = Field(default="", description="Long-lived access token")
    entity_ids: str = Field(
        default="", description="Comma-separated entity IDs to monitor"
    )
    poll_interval: float = Field(
        default=30.0, description="Seconds between state polls"
    )


class HomeAssistantStateInput(FuserInput[HomeAssistantInputConfig, Optional[list]]):
    """
    Home Assistant state input that polls device states and reports changes.

    Monitors smart home device states and provides updates to the LLM
    when states change, enabling context-aware robot responses.
    """

    def __init__(self, config: HomeAssistantInputConfig):
        """
        Initialize the Home Assistant state input.

        Parameters
        ----------
        config : HomeAssistantInputConfig
            Configuration for the state input.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Home Assistant Device States"

        self.base_url = config.base_url.rstrip("/")
        self.token = config.token
        self.entity_ids = [e.strip() for e in config.entity_ids.split(",") if e.strip()]
        self.poll_interval = config.poll_interval

        self._last_poll_time: float = 0
        self._last_states: dict = {}

        if not self.base_url:
            logging.warning("HomeAssistantStateInput: base_url not provided")
        if not self.token:
            logging.warning("HomeAssistantStateInput: token not provided")
        if not self.entity_ids:
            logging.warning("HomeAssistantStateInput: no entity_ids configured")

    def _get_headers(self) -> dict:
        """Build authorization headers."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _fetch_state(self, entity_id: str) -> Optional[dict]:
        """
        Fetch the current state of a single entity from Home Assistant.

        Parameters
        ----------
        entity_id : str
            The entity ID to fetch.

        Returns
        -------
        Optional[dict]
            State data or None on failure.
        """
        if not self.base_url or not self.token:
            return None

        url = f"{self.base_url}/api/states/{entity_id}"

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self._get_headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logging.error(
                            f"HomeAssistantStateInput: error fetching {entity_id} "
                            f"status={response.status}"
                        )
                        return None
        except asyncio.TimeoutError:
            logging.error(f"HomeAssistantStateInput: timeout fetching {entity_id}")
            return None
        except aiohttp.ClientError as e:
            logging.error(
                f"HomeAssistantStateInput: network error fetching {entity_id}: {e}"
            )
            return None
        except Exception as e:
            logging.error(
                f"HomeAssistantStateInput: unexpected error fetching {entity_id}: {e}"
            )
            return None

    async def _poll(self) -> Optional[list]:
        """
        Poll Home Assistant for device states at configured interval.

        Returns
        -------
        Optional[list]
            List of state dicts when poll interval elapsed, None otherwise.
        """
        current_time = time.time()

        if current_time - self._last_poll_time < self.poll_interval:
            await asyncio.sleep(1.0)
            return None

        self._last_poll_time = current_time
        await asyncio.sleep(1.0)

        if not self.entity_ids:
            return None

        states = []
        for entity_id in self.entity_ids:
            state = await self._fetch_state(entity_id)
            if state is not None:
                states.append(state)

        return states if states else None

    def _format_state(self, state: dict) -> str:
        """
        Format a single entity state into human-readable text.

        Parameters
        ----------
        state : dict
            Raw state data from Home Assistant API.

        Returns
        -------
        str
            Human-readable state description.
        """
        entity_id = state.get("entity_id", "unknown")
        current_state = state.get("state", "unknown")
        attributes = state.get("attributes", {})
        friendly_name = attributes.get("friendly_name", entity_id)

        parts = [f"{friendly_name} ({entity_id}) is {current_state}"]

        if "brightness" in attributes and attributes["brightness"] is not None:
            brightness_pct = round(attributes["brightness"] / 255 * 100)
            parts.append(f"brightness {brightness_pct}%")

        if "color_name" in attributes:
            parts.append(f"color {attributes['color_name']}")

        if "temperature" in attributes:
            parts.append(f"temperature {attributes['temperature']}°C")

        if "current_temperature" in attributes:
            parts.append(f"current temperature {attributes['current_temperature']}°C")

        return ", ".join(parts)

    async def _raw_to_text(self, raw_input: Optional[list]) -> Optional[Message]:
        """
        Convert raw state list to human-readable message, only when states change.

        Parameters
        ----------
        raw_input : Optional[list]
            List of state dicts from Home Assistant.

        Returns
        -------
        Optional[Message]
            Formatted message if states changed, None otherwise.
        """
        if raw_input is None:
            return None

        changed = []
        for state in raw_input:
            entity_id = state.get("entity_id", "")
            current_state = state.get("state", "")
            last_known = self._last_states.get(entity_id)

            if last_known != current_state:
                changed.append(state)
                self._last_states[entity_id] = current_state

        if not changed:
            return None

        lines = [self._format_state(s) for s in changed]
        message = "Smart home device updates: " + "; ".join(lines)
        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Optional[list]):
        """
        Update message buffer with processed state data.

        Parameters
        ----------
        raw_input : Optional[list]
            Raw state list to process.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string for LLM or None if buffer is empty.
        """
        if not self.messages:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM,
            latest_message.message,
            latest_message.timestamp,
        )
        self.messages = []

        return result
