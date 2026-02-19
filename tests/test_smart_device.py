"""
Tests for SmartDeviceInput and SmartDeviceAction
Bounty #366 — https://github.com/OpenMind/OM1/issues/366

Run with:
    uv run pytest tests/test_smart_device.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inputs.plugins.smart_device import SmartDeviceInput, DeviceState
from actions.smart_device.action import SmartDeviceAction


SAMPLE_LIGHT = {
    "entity_id": "light.living_room",
    "state": "on",
    "attributes": {
        "friendly_name": "Living Room Light",
        "brightness": 128,
    },
    "last_changed": "2025-01-01T12:00:00Z",
}


def make_plugin(entity_ids=None):
    return SmartDeviceInput(config={
        "ha_url":        "http://ha.local:8123",
        "ha_token":      "test_token",
        "entity_ids":    entity_ids or ["light.living_room"],
        "poll_interval": 999,
    })


class TestStateToText:
    def _ds(self, entity_id, state, attrs=None):
        return DeviceState(
            entity_id=entity_id,
            friendly_name=attrs.get("friendly_name", entity_id) if attrs else entity_id,
            state=state,
            attributes=attrs or {},
            last_changed="",
        )

    def test_light_on(self):
        ds = self._ds("light.x", "on", {"friendly_name": "Kitchen", "brightness": 255})
        text = SmartDeviceInput._state_to_text(ds)
        assert "ON" in text
        assert "100%" in text

    def test_light_off(self):
        ds = self._ds("light.x", "off", {"friendly_name": "Bedroom Light"})
        text = SmartDeviceInput._state_to_text(ds)
        assert "OFF" in text

    def test_climate(self):
        ds = self._ds("climate.t", "heat", {
            "friendly_name": "Thermostat",
            "current_temperature": 20,
            "temperature": 22,
            "hvac_mode": "heat",
        })
        text = SmartDeviceInput._state_to_text(ds)
        assert "20" in text and "22" in text

    def test_binary_sensor_closed(self):
        ds = self._ds("binary_sensor.door", "off", {"friendly_name": "Door"})
        assert "closed" in SmartDeviceInput._state_to_text(ds).lower()

    def test_binary_sensor_open(self):
        ds = self._ds("binary_sensor.door", "on", {"friendly_name": "Door"})
        assert "open" in SmartDeviceInput._state_to_text(ds).lower()

    def test_lock(self):
        ds = self._ds("lock.front", "locked", {"friendly_name": "Front Lock"})
        assert "locked" in SmartDeviceInput._state_to_text(ds).lower()

    def test_sensor_with_unit(self):
        ds = self._ds("sensor.temp", "21.5", {
            "friendly_name": "Temp", "unit_of_measurement": "°C"
        })
        assert "21.5°C" in SmartDeviceInput._state_to_text(ds)

    def test_media_player(self):
        ds = self._ds("media_player.s", "playing", {
            "friendly_name": "Speaker", "media_title": "Lo-Fi Beats"
        })
        assert "Lo-Fi Beats" in SmartDeviceInput._state_to_text(ds)


class TestSmartDeviceInput:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_states(self):
        plugin = make_plugin([])
        plugin._last_poll = 1e18
        assert await plugin.raw() is None

    @pytest.mark.asyncio
    async def test_returns_summary_from_cached_state(self):
        plugin = make_plugin()
        plugin._last_poll = 1e18
        plugin._states["light.living_room"] = DeviceState(
            entity_id="light.living_room",
            friendly_name="Living Room Light",
            state="off",
            attributes={},
            last_changed="",
        )
        result = await plugin.raw()
        assert result is not None
        assert "SMART HOME STATUS" in result

    @pytest.mark.asyncio
    async def test_call_service_success(self):
        plugin = make_plugin()
        mock_ctx = MagicMock()
        ok_resp = MagicMock()
        ok_resp.status = 200
        mock_ctx.__aenter__ = AsyncMock(return_value=ok_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session.closed = False
        plugin._session = mock_session
        assert await plugin.call_service("light", "turn_on", {"entity_id": "light.x"}) is True

    @pytest.mark.asyncio
    async def test_cleanup(self):
        plugin = make_plugin()
        mock_session = AsyncMock()
        mock_session.closed = False
        plugin._session = mock_session
        await plugin.cleanup()
        mock_session.close.assert_awaited_once()


class TestSmartDeviceAction:
    def make_action(self):
        mock_input = AsyncMock()
        mock_input.call_service = AsyncMock(return_value=True)
        return SmartDeviceAction(input_plugin=mock_input), mock_input

    @pytest.mark.asyncio
    async def test_turn_on(self):
        action, mock_input = self.make_action()
        assert await action.act({"command": "turn_on", "entity_id": "light.x"}) is True
        mock_input.call_service.assert_awaited_once_with(
            "homeassistant", "turn_on", {"entity_id": "light.x"}
        )

    @pytest.mark.asyncio
    async def test_turn_off(self):
        action, mock_input = self.make_action()
        assert await action.act({"command": "turn_off", "entity_id": "switch.coffee"}) is True

    @pytest.mark.asyncio
    async def test_set_temperature(self):
        action, mock_input = self.make_action()
        await action.act({"command": "set_temperature", "entity_id": "climate.t", "value": 23})
        _, _, data = mock_input.call_service.call_args[0]
        assert data["temperature"] == 23.0

    @pytest.mark.asyncio
    async def test_set_brightness(self):
        action, mock_input = self.make_action()
        await action.act({"command": "set_brightness", "entity_id": "light.x", "value": 50})
        _, _, data = mock_input.call_service.call_args[0]
        assert data["brightness_pct"] == 50

    @pytest.mark.asyncio
    async def test_lock(self):
        action, mock_input = self.make_action()
        assert await action.act({"command": "lock", "entity_id": "lock.front"}) is True

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        action, mock_input = self.make_action()
        assert await action.act({"command": "fly", "entity_id": "light.x"}) is False
        mock_input.call_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_entity_id(self):
        action, mock_input = self.make_action()
        assert await action.act({"command": "turn_on", "entity_id": ""}) is False

    @pytest.mark.asyncio
    async def test_no_input_plugin(self):
        action = SmartDeviceAction(input_plugin=None)
        assert await action.act({"command": "turn_on", "entity_id": "light.x"}) is False