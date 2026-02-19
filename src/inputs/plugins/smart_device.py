"""
Smart Device Communication Input Plugin for OM1
Bounty #366 — https://github.com/OpenMind/OM1/issues/366
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SmartDeviceConfig:
    ha_url: str = "http://homeassistant.local:8123"
    ha_token: str = ""
    entity_ids: list[str] = field(default_factory=list)
    poll_interval: float = 10.0
    request_timeout: float = 5.0
    name_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class DeviceState:
    entity_id: str
    friendly_name: str
    state: str
    attributes: dict[str, Any]
    last_changed: str


class SmartDeviceInput:
    PLUGIN_NAME = "SmartDeviceInput"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cfg = SmartDeviceConfig(
            ha_url=cfg.get("ha_url", "http://homeassistant.local:8123").rstrip("/"),
            ha_token=cfg.get("ha_token", ""),
            entity_ids=cfg.get("entity_ids", []),
            poll_interval=float(cfg.get("poll_interval", 10.0)),
            request_timeout=float(cfg.get("request_timeout", 5.0)),
            name_overrides=cfg.get("name_overrides", {}),
        )
        self._states: dict[str, DeviceState] = {}
        self._last_poll: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._cfg.ha_token}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout)
            self._session = aiohttp.ClientSession(
                headers=self._headers(), timeout=timeout
            )
        return self._session

    async def _fetch_state(self, entity_id: str) -> Optional[DeviceState]:
        url = f"{self._cfg.ha_url}/api/states/{entity_id}"
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    friendly_name = (
                        self._cfg.name_overrides.get(entity_id)
                        or data.get("attributes", {}).get("friendly_name", entity_id)
                    )
                    return DeviceState(
                        entity_id=entity_id,
                        friendly_name=friendly_name,
                        state=data.get("state", "unknown"),
                        attributes=data.get("attributes", {}),
                        last_changed=data.get("last_changed", ""),
                    )
                else:
                    logger.warning("[SmartDevice] HA returned %s for '%s'", resp.status, entity_id)
        except asyncio.TimeoutError:
            logger.error("[SmartDevice] Timeout fetching '%s'", entity_id)
        except Exception as exc:
            logger.error("[SmartDevice] Error for '%s': %s", entity_id, exc)
        return None

    async def _poll_all(self) -> None:
        tasks = [self._fetch_state(eid) for eid in self._cfg.entity_ids]
        results = await asyncio.gather(*tasks)
        async with self._lock:
            for result in results:
                if result is not None:
                    self._states[result.entity_id] = result

    @staticmethod
    def _state_to_text(ds: DeviceState) -> str:
        domain = ds.entity_id.split(".")[0]
        name = ds.friendly_name
        state = ds.state
        attrs = ds.attributes

        if domain == "light":
            if state == "on":
                brightness_pct = round(attrs.get("brightness", 255) / 255 * 100)
                return f"{name} is ON at {brightness_pct}% brightness."
            return f"{name} is OFF."

        if domain == "climate":
            current = attrs.get("current_temperature", "?")
            target = attrs.get("temperature", "?")
            mode = attrs.get("hvac_mode", state)
            return f"{name}: current {current}°, target {target}°, mode '{mode}'."

        if domain == "binary_sensor":
            human = "open" if state == "on" else "closed"
            return f"{name} is {human}."

        if domain == "lock":
            return f"{name} is {state}."

        if domain == "sensor":
            unit = attrs.get("unit_of_measurement", "")
            return f"{name} reads {state}{unit}."

        if domain == "media_player":
            media_title = attrs.get("media_title", "")
            detail = f" playing '{media_title}'" if media_title else ""
            return f"{name} is {state}{detail}."

        if domain == "switch":
            return f"{name} is {'on' if state == 'on' else 'off'}."

        return f"{name} ({ds.entity_id}) is '{state}'."

    async def _get_data(self) -> Optional[str]:
        now = time.monotonic()
        if now - self._last_poll >= self._cfg.poll_interval:
            await self._poll_all()
            self._last_poll = now
        async with self._lock:
            if not self._states:
                return None
            lines = [self._state_to_text(ds) for ds in self._states.values()]
        return "SMART HOME STATUS:\n" + "\n".join(f"  - {l}" for l in lines)

    async def raw(self) -> Optional[str]:
        return await self._get_data()

    async def cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def call_service(self, domain: str, service: str, service_data: dict[str, Any]) -> bool:
        url = f"{self._cfg.ha_url}/api/services/{domain}/{service}"
        session = await self._get_session()
        try:
            async with session.post(url, json=service_data) as resp:
                if resp.status in (200, 201):
                    return True
                body = await resp.text()
                logger.error("[SmartDevice] service %s.%s failed %s: %s", domain, service, resp.status, body)
        except Exception as exc:
            logger.error("[SmartDevice] call_service error: %s", exc)
        return False