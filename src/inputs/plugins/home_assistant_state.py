import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.home_assistant_provider import HomeAssistantProvider

logger = logging.getLogger(__name__)


class HomeAssistantStateConfig(SensorConfig):
    """
    Configuration for the Home Assistant state input.

    Parameters
    ----------
    base_url : str
        Base URL of the Home Assistant instance.
    token_env : str
        Environment variable name for the HA access token.
    token : str
        Direct access token (fallback if env var not set).
    entities : List[str]
        List of entity IDs to monitor.
    poll_interval : float
        Polling interval in seconds.
    verify_ssl : bool
        Whether to verify SSL certificates.
    """

    base_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Base URL of the Home Assistant instance",
    )
    token_env: str = Field(
        default="HOME_ASSISTANT_TOKEN",
        description="Environment variable name for the HA access token",
    )
    token: str = Field(
        default="",
        description="Direct access token (fallback if env var not set)",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of entity IDs to monitor",
    )
    poll_interval: float = Field(
        default=10.0,
        description="Polling interval in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )


class HomeAssistantStateInput(
    FuserInput[HomeAssistantStateConfig, Optional[Dict[str, Any]]]
):
    """
    Input plugin that polls Home Assistant entity states.

    Periodically fetches entity states from Home Assistant and reports
    changes to the LLM via the fuser system.
    """

    def __init__(self, config: HomeAssistantStateConfig):
        """
        Initialize the Home Assistant state input.

        Parameters
        ----------
        config : HomeAssistantStateConfig
            Configuration for the input plugin.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Home Status"
        self.messages: List[Message] = []
        self.last_states: Dict[str, str] = {}

        if not config.entities:
            logger.warning(
                "HomeAssistantStateInput: No entities configured. "
                "Add entity IDs to the 'entities' list in the config."
            )

        self.provider = HomeAssistantProvider(
            base_url=config.base_url,
            token=config.token,
            token_env=config.token_env,
            verify_ssl=config.verify_ssl,
        )

    async def _poll(self) -> Optional[Dict[str, Any]]:
        """
        Poll Home Assistant for entity states.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary mapping entity_id to state data, or None on error.
        """
        await asyncio.sleep(self.config.poll_interval)

        if not self.config.entities:
            return None

        try:
            states = await self.provider.get_states(self.config.entities)
            return {s["entity_id"]: s for s in states}
        except Exception as e:
            logger.error(f"HomeAssistantStateInput: Error polling states: {e}")
            return None

    def _format_entity_state(self, state_data: Dict[str, Any]) -> str:
        """
        Format a single entity state into a human-readable string.

        Parameters
        ----------
        state_data : Dict[str, Any]
            The state object from Home Assistant.

        Returns
        -------
        str
            Formatted state string.
        """
        attributes = state_data.get("attributes", {})
        friendly_name = attributes.get(
            "friendly_name", state_data.get("entity_id", "unknown")
        )
        state = state_data.get("state", "unknown")
        unit = attributes.get("unit_of_measurement", "")

        parts = [f"{friendly_name}: {state}"]
        if unit:
            parts[0] += f" {unit}"

        entity_id = state_data.get("entity_id", "")
        domain = entity_id.split(".", 1)[0] if "." in entity_id else ""

        if domain == "light" and state == "on":
            brightness = attributes.get("brightness")
            if brightness is not None:
                brightness_pct = round(brightness / 255 * 100)
                parts.append(f"brightness {brightness_pct}%")

        if domain == "climate":
            current_temp = attributes.get("current_temperature")
            if current_temp is not None:
                parts.append(f"current temperature {current_temp}")

        return ", ".join(parts)

    async def _raw_to_text(
        self, raw_input: Optional[Dict[str, Any]]
    ) -> Optional[Message]:
        """
        Convert raw state data to a text message, reporting only changes.

        Parameters
        ----------
        raw_input : Optional[Dict[str, Any]]
            Dictionary of entity states from _poll.

        Returns
        -------
        Optional[Message]
            A message with changed states, or None if no changes.
        """
        if raw_input is None:
            return None

        changed_lines = []
        for entity_id, state_data in raw_input.items():
            state_value = state_data.get("state", "")
            previous = self.last_states.get(entity_id)

            if previous != state_value:
                self.last_states[entity_id] = state_value
                changed_lines.append(self._format_entity_state(state_data))

        if not changed_lines:
            return None

        text = "\n".join(changed_lines)
        return Message(timestamp=time.time(), message=text)

    async def raw_to_text(self, raw_input: Optional[Dict[str, Any]]) -> None:
        """
        Process polled state data and buffer changed states.

        Parameters
        ----------
        raw_input : Optional[Dict[str, Any]]
            Dictionary of entity states from _poll.
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
            Formatted string of buffer contents, or None if empty.
        """
        if not self.messages:
            return None

        latest_message = self.messages[-1]

        result = (
            f"INPUT: {self.descriptor_for_LLM}\n"
            f"// START\n"
            f"{latest_message.message}\n"
            f"// END"
        )

        self.messages.clear()
        return result
