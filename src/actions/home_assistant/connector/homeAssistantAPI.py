import json
import logging
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantInput


class HomeAssistantAPIConfig(ActionConfig):
    """Configuration for HomeAssistantAPIConnector."""

    base_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant base URL, e.g. http://<ip>:8123",
    )
    token: str = Field(
        default="",
        description="Home Assistant long-lived access token (recommended to use env var)",
    )
    token_env: str = Field(
        default="HOME_ASSISTANT_TOKEN",
        description="Environment variable name to read token from if token is empty",
    )

    # Friendly alias -> entity_id mapping (avoids the LLM guessing entity_ids)
    devices: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of device alias to Home Assistant entity_id",
    )

    timeout_seconds: float = Field(default=10.0, description="HTTP timeout seconds")
    verify_ssl: bool = Field(default=True, description="Verify TLS certs")


class HomeAssistantAPIConnector(ActionConnector[HomeAssistantAPIConfig, HomeAssistantInput]):
    """Connector for Home Assistant REST API service calls."""

    def _get_token(self) -> str:
        token = (self.config.token or "").strip()
        if token:
            return token

        env_key = (self.config.token_env or "").strip()
        if not env_key:
            return ""

        import os

        return (os.environ.get(env_key) or "").strip()

    def _resolve_entity_id(self, device_alias: str) -> str:
        entity_id = (self.config.devices or {}).get(device_alias)
        if not entity_id:
            raise ValueError(
                f"Unknown Home Assistant device alias '{device_alias}'. "
                f"Add it to config.devices (alias -> entity_id)."
            )
        return entity_id

    @staticmethod
    def _split_domain(entity_id: str) -> Tuple[str, str]:
        if "." not in entity_id:
            raise ValueError(
                f"Invalid entity_id '{entity_id}'. Expected '<domain>.<entity>'."
            )
        domain, _ = entity_id.split(".", 1)
        return domain, entity_id

    @staticmethod
    def _command_to_service(domain: str, command: str, value: Optional[float]) -> Tuple[str, Dict[str, Any]]:
        c = (command or "").strip().lower()

        if c in {"on", "off", "toggle"}:
            return {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[c], {}

        if c == "set":
            if value is None:
                raise ValueError("'set' requires a numeric value")

            if domain == "light":
                return "turn_on", {"brightness_pct": float(value)}
            if domain == "climate":
                return "set_temperature", {"temperature": float(value)}
            if domain in {"input_number", "number"}:
                return "set_value", {"value": float(value)}

            # fallback
            return "set_value", {"value": float(value)}

        raise ValueError("Unsupported command. Use on/off/toggle/set")

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        token = self._get_token()
        if not token:
            logging.error(
                "Home Assistant token missing. Set config.token or env var HOME_ASSISTANT_TOKEN."
            )
            return

        # The orchestrator passes action.value to this connector as output_interface.action.
        # We expect it to be JSON for structured arguments.
        try:
            payload_in = json.loads(output_interface.action or "{}")
        except Exception as e:
            raise ValueError(
                "HomeAssistant action expects JSON string in `action`, e.g. "
                "{\"device\":\"living_room_light\",\"command\":\"on\"}"
            ) from e

        if not isinstance(payload_in, dict):
            raise ValueError("HomeAssistant action JSON must be an object")

        device = payload_in.get("device")
        command = payload_in.get("command")
        value = payload_in.get("value", None)

        if not device or not command:
            raise ValueError("Missing required keys: device, command")

        entity_id = self._resolve_entity_id(str(device))
        domain, entity_id = self._split_domain(entity_id)
        service, extra = self._command_to_service(domain, str(command), None if value is None else float(value))

        url = f"{self.config.base_url.rstrip('/')}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload_out: Dict[str, Any] = {"entity_id": entity_id, **extra}

        timeout = aiohttp.ClientTimeout(total=float(self.config.timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload_out, headers=headers, ssl=bool(self.config.verify_ssl)) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Home Assistant call failed: {resp.status} {resp.reason}: {text[:500]}"
                    )
