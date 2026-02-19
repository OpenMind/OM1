"""
Smart Device Action Plugin for OM1
Bounty #366 — https://github.com/OpenMind/OM1/issues/366
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SmartDeviceCommand:
    command: str
    entity_id: str
    value: Optional[Any] = None


_COMMAND_MAP: dict[str, tuple[str, str]] = {
    "turn_on":         ("homeassistant", "turn_on"),
    "turn_off":        ("homeassistant", "turn_off"),
    "toggle":          ("homeassistant", "toggle"),
    "lock":            ("lock",          "lock"),
    "unlock":          ("lock",          "unlock"),
    "set_temperature": ("climate",       "set_temperature"),
    "set_brightness":  ("light",         "turn_on"),
    "play_media":      ("media_player",  "play_media"),
}


class SmartDeviceAction:
    PLUGIN_NAME = "SmartDeviceAction"

    def __init__(self, input_plugin=None, config: Optional[dict] = None) -> None:
        self._input = input_plugin
        self._config = config or {}

    def _build_service_data(self, cmd: SmartDeviceCommand) -> dict[str, Any]:
        data: dict[str, Any] = {"entity_id": cmd.entity_id}

        if cmd.command == "set_temperature" and cmd.value is not None:
            data["temperature"] = float(cmd.value)

        elif cmd.command == "set_brightness" and cmd.value is not None:
            data["brightness_pct"] = max(0, min(100, int(cmd.value)))

        elif cmd.command == "play_media" and cmd.value is not None:
            if isinstance(cmd.value, dict):
                data.update(cmd.value)
            else:
                data["media_content_id"] = str(cmd.value)
                data["media_content_type"] = "music"

        return data

    async def act(self, command_dict: dict[str, Any]) -> bool:
        try:
            cmd = SmartDeviceCommand(
                command=command_dict.get("command", "").lower(),
                entity_id=command_dict.get("entity_id", ""),
                value=command_dict.get("value"),
            )
        except Exception as exc:
            logger.error("[SmartDeviceAction] Failed to parse command: %s", exc)
            return False

        if not cmd.command or not cmd.entity_id:
            logger.warning("[SmartDeviceAction] Missing command or entity_id.")
            return False

        if cmd.command not in _COMMAND_MAP:
            logger.warning(
                "[SmartDeviceAction] Unknown command '%s'. Supported: %s",
                cmd.command,
                list(_COMMAND_MAP.keys()),
            )
            return False

        domain, service = _COMMAND_MAP[cmd.command]
        service_data = self._build_service_data(cmd)

        if self._input is None:
            logger.error("[SmartDeviceAction] No input plugin reference.")
            return False

        success = await self._input.call_service(domain, service, service_data)
        if success:
            logger.info(
                "[SmartDeviceAction] ✓ %s → %s.%s %s",
                cmd.command, domain, service, service_data,
            )
        return success