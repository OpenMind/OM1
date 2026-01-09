import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantInput
from providers.io_provider import IOProvider


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for Home Assistant connector.

    Parameters
    ----------
    ha_url : str
        Home Assistant URL (e.g., "http://homeassistant.local:8123").
    access_token : str
        Long-lived access token from Home Assistant.
    switch_entity_id : Optional[str]
        Entity ID for switch control (e.g., "switch.tapo_plug").
    climate_entity_id : Optional[str]
        Entity ID for climate/thermostat control (e.g., "climate.lg_ac").
    light_entity_id : Optional[str]
        Entity ID for light control (e.g., "light.living_room").
    """

    ha_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant URL",
    )
    access_token: Optional[str] = Field(
        default=None,
        description="Long-lived access token from Home Assistant",
    )
    switch_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for switch control",
    )
    climate_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for climate/thermostat control",
    )
    light_entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for light control",
    )


def extract_temperature(text: str) -> Optional[float]:
    """Extract temperature value from text like 'set temperature 24' or '24 degrees'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|celsius|c)?", text.lower())
    if match:
        temp = float(match.group(1))
        if 16 <= temp <= 30:
            return temp
    return None


class HomeAssistantConnector(ActionConnector[HomeAssistantConfig, HomeAssistantInput]):
    """
    Connector for controlling smart home devices via Home Assistant REST API.
    """

    _last_action: Optional[str] = None

    def __init__(self, config: HomeAssistantConfig):
        super().__init__(config)

        self.io_provider = IOProvider()

        self.ha_url = config.ha_url
        self.access_token = config.access_token
        self.switch_entity_id = config.switch_entity_id
        self.climate_entity_id = config.climate_entity_id
        self.light_entity_id = config.light_entity_id

        self._session: Optional[aiohttp.ClientSession] = None

        if not self.access_token:
            logging.error(
                "HomeAssistantConnector: No access_token provided. "
                "Create one in Home Assistant Profile → Long-Lived Access Tokens"
            )

        logging.info(
            f"\033[94mHomeAssistantConnector: Initialized ({self.ha_url})\033[0m"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers for Home Assistant API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Call a Home Assistant service."""
        # Check current state to avoid duplicate commands
        current_state = await self.get_entity_state(entity_id)
        if current_state:
            # Switch: skip if already in desired state
            if domain == "switch":
                if service == "turn_off" and current_state == "off":
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already off, skipping\033[0m"
                    )
                    return True
                if service == "turn_on" and current_state == "on":
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already on, skipping\033[0m"
                    )
                    return True
            # Climate: skip if already in desired hvac_mode
            if domain == "climate" and service == "set_hvac_mode":
                desired_mode = data.get("hvac_mode") if data else None
                if desired_mode and current_state == desired_mode:
                    logging.info(
                        f"\033[90mHomeAssistant: {entity_id} already {desired_mode}, skipping\033[0m"
                    )
                    return True

        url = f"{self.ha_url}/api/services/{domain}/{service}"
        session = await self._get_session()

        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        try:
            async with session.post(
                url, headers=self._get_headers(), json=payload, timeout=10
            ) as resp:
                if resp.status == 200:
                    # Different colors per domain: switch=magenta, climate=cyan, light=yellow
                    color = {"switch": "95", "climate": "96", "light": "93"}.get(
                        domain, "96"
                    )
                    logging.info(
                        f"\033[{color}mHomeAssistant: {domain}.{service} → {entity_id}\033[0m"
                    )
                    return True
                else:
                    text = await resp.text()
                    logging.error(
                        f"\033[91mHomeAssistant: API error {resp.status}: {text}\033[0m"
                    )
        except asyncio.TimeoutError:
            logging.error("\033[91mHomeAssistant: Request timed out\033[0m")
        except aiohttp.ClientError as e:
            logging.error(f"\033[91mHomeAssistant: Connection error: {e}\033[0m")
        except Exception as e:
            logging.error(f"\033[91mHomeAssistant: Error: {e}\033[0m")

        return False

    def _parse_action(self, action_text: str) -> List[Dict[str, Any]]:
        """Parse action text and return list of commands."""
        text = action_text.lower().strip()
        commands = []

        if text in ["idle", "nothing", "do nothing", "no action"]:
            return []

        # Switch/plug commands
        if self.switch_entity_id:
            if any(
                x in text
                for x in [
                    "turn on switch",
                    "switch on",
                    "plug on",
                    "turn on plug",
                    "turn on fan",
                    "fan on",
                ]
            ):
                commands.append(
                    {
                        "domain": "switch",
                        "service": "turn_on",
                        "entity_id": self.switch_entity_id,
                    }
                )
            elif any(
                x in text
                for x in [
                    "turn off switch",
                    "switch off",
                    "plug off",
                    "turn off plug",
                    "turn off fan",
                    "fan off",
                ]
            ):
                commands.append(
                    {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": self.switch_entity_id,
                    }
                )

        # Light commands
        if self.light_entity_id:
            if any(
                x in text
                for x in ["turn on light", "light on", "turn on lamp", "lamp on"]
            ):
                commands.append(
                    {
                        "domain": "light",
                        "service": "turn_on",
                        "entity_id": self.light_entity_id,
                    }
                )
            elif any(
                x in text
                for x in ["turn off light", "light off", "turn off lamp", "lamp off"]
            ):
                commands.append(
                    {
                        "domain": "light",
                        "service": "turn_off",
                        "entity_id": self.light_entity_id,
                    }
                )

        # Climate/thermostat commands
        if self.climate_entity_id:
            # Temperature setting
            temp = extract_temperature(text)
            if temp is not None:
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_temperature",
                        "entity_id": self.climate_entity_id,
                        "data": {"temperature": temp},
                    }
                )

            # AC on/off control
            # turn on AC → set temperature to 24 (this turns on AC automatically)
            if any(
                x in text
                for x in ["turn on ac", "ac on", "turn on climate", "turn on hvac"]
            ):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_temperature",
                        "entity_id": self.climate_entity_id,
                        "data": {"temperature": 24},
                    }
                )
            # turn off AC → set_hvac_mode off
            elif any(
                x in text
                for x in ["turn off ac", "ac off", "turn off climate", "turn off hvac"]
            ):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "off"},
                    }
                )
            # Specific HVAC modes
            elif any(x in text for x in ["cool mode", "cooling", "set to cool"]):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "cool"},
                    }
                )
            elif any(x in text for x in ["heat mode", "heating", "set to heat"]):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "heat"},
                    }
                )
            elif any(x in text for x in ["dry mode", "dehumidify", "set to dry"]):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "dry"},
                    }
                )
            elif any(x in text for x in ["auto mode", "automatic", "set to auto"]):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "auto"},
                    }
                )
            elif any(x in text for x in ["fan only", "fan mode", "set to fan"]):
                commands.append(
                    {
                        "domain": "climate",
                        "service": "set_hvac_mode",
                        "entity_id": self.climate_entity_id,
                        "data": {"hvac_mode": "fan_only"},
                    }
                )

        return commands

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute Home Assistant actions based on LLM decision.
        Skips if same action was already executed.
        """
        action_text = output_interface.action

        if not action_text:
            return

        normalized = action_text.lower().strip()

        if normalized in ["idle", "nothing", "do nothing", "no action"]:
            logging.info("\033[93mHomeAssistant: LLM decided → idle\033[0m")
            return

        if normalized == HomeAssistantConnector._last_action:
            logging.info("\033[90mHomeAssistant: Same action, skipping\033[0m")
            return

        logging.info(f"\033[92mHomeAssistant: LLM decided → {normalized}\033[0m")

        commands = self._parse_action(normalized)

        for cmd in commands:
            await self._call_service(
                domain=cmd["domain"],
                service=cmd["service"],
                entity_id=cmd["entity_id"],
                data=cmd.get("data"),
            )

        HomeAssistantConnector._last_action = normalized

    async def get_entity_state(self, entity_id: str) -> Optional[str]:
        """Get current state of a specific entity."""
        url = f"{self.ha_url}/api/states/{entity_id}"
        session = await self._get_session()

        try:
            async with session.get(url, headers=self._get_headers(), timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("state")
        except Exception as e:
            logging.debug(f"HomeAssistant: Error getting state for {entity_id}: {e}")

        return None

    async def get_states(self) -> Optional[List[Dict[str, Any]]]:
        """Get all entity states from Home Assistant."""
        url = f"{self.ha_url}/api/states"
        session = await self._get_session()

        try:
            async with session.get(
                url, headers=self._get_headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logging.error(f"\033[91mHomeAssistant: Error getting states: {e}\033[0m")

        return None

    def __del__(self):
        """Cleanup session on destruction."""
        if self._session and not self._session.closed:
            try:
                asyncio.get_event_loop().create_task(self._session.close())
            except RuntimeError:
                pass
