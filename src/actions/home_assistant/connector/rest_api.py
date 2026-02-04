import os
from typing import Any, Dict, Optional, Tuple

import aiohttp

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantControlInput


class HomeAssistantConfig(ActionConfig):
    """Configuration for Home Assistant control."""

    base_url: str = "http://homeassistant.local:8123"

    # Prefer env var so configs can be committed safely
    token_env: str = "HOME_ASSISTANT_TOKEN"
    token: str = ""

    # Map friendly device names to entity_ids.
    # Example: {"living_room_light": "light.living_room"}
    devices: Dict[str, str] = {}

    timeout_seconds: float = 10.0
    verify_ssl: bool = True


class HomeAssistantRESTConnector(ActionConnector[HomeAssistantConfig, HomeAssistantControlInput]):
    """Calls Home Assistant REST API to control entities."""

    def _get_token(self) -> str:
        token = (self.config.token or "").strip()
        if token:
            return token
        env_key = (self.config.token_env or "").strip()
        if env_key:
            return (os.environ.get(env_key) or "").strip()
        return ""

    def _resolve_entity(self, device: str) -> str:
        entity_id = (self.config.devices or {}).get(device)
        if not entity_id:
            raise ValueError(
                f"Unknown Home Assistant device alias '{device}'. "
                f"Add it to actions.home_assistant.config.devices."
            )
        return entity_id

    @staticmethod
    def _split_domain_entity(entity_id: str) -> Tuple[str, str]:
        if "." not in entity_id:
            raise ValueError(
                f"Invalid entity_id '{entity_id}'. Expected format '<domain>.<entity>'."
            )
        domain, _ = entity_id.split(".", 1)
        return domain, entity_id

    def _command_to_service(
        self, domain: str, command: str, value: Optional[float]
    ) -> Tuple[str, Dict[str, Any]]:
        c = command.strip().lower()

        if c in {"on", "off", "toggle"}:
            return (
                {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[c],
                {},
            )

        if c == "set":
            if value is None:
                raise ValueError("'set' command requires a numeric value")

            # Small opinionated defaults that work well in practice.
            if domain == "light":
                # Interpret value as brightness percentage.
                return "turn_on", {"brightness_pct": float(value)}

            if domain == "climate":
                return "set_temperature", {"temperature": float(value)}

            if domain in {"input_number", "number"}:
                return "set_value", {"value": float(value)}

            # Fallback: try a generic 'set_value'
            return "set_value", {"value": float(value)}

        raise ValueError(f"Unsupported command '{command}'. Use on/off/toggle/set")

    async def connect(self, output_interface: HomeAssistantControlInput) -> None:
        token = self._get_token()
        if not token:
            raise ValueError(
                "Missing Home Assistant token. Set config.token or env var HOME_ASSISTANT_TOKEN."
            )

        entity_id = self._resolve_entity(output_interface.device)
        domain, entity_id = self._split_domain_entity(entity_id)
        service, extra = self._command_to_service(
            domain, output_interface.command, output_interface.value
        )

        url = f"{self.config.base_url.rstrip('/')}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"entity_id": entity_id, **extra}

        timeout = aiohttp.ClientTimeout(total=float(self.config.timeout_seconds))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                ssl=bool(self.config.verify_ssl),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Home Assistant call failed: {resp.status} {resp.reason}: {text[:500]}"
                    )
