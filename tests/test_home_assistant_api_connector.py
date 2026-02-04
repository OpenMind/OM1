import json

import pytest

from actions.home_assistant.connector.homeAssistantAPI import (
    HomeAssistantAPIConfig,
    HomeAssistantAPIConnector,
)
from actions.home_assistant.interface import HomeAssistantInput


class _FakeResponse:
    def __init__(self, status=200, reason="OK", text="[]"):
        self.status = status
        self.reason = reason
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None, ssl=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "ssl": ssl})
        return _FakeResponse()


@pytest.mark.asyncio
async def test_turn_on_light_builds_expected_call(monkeypatch):
    cfg = HomeAssistantAPIConfig(
        base_url="http://localhost:8123",
        token="abc",
        devices={"lamp": "light.living_room"},
        verify_ssl=False,
    )
    c = HomeAssistantAPIConnector(cfg)

    fake = _FakeSession()

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout=None: fake)

    action = json.dumps({"device": "lamp", "command": "on"})
    await c.connect(HomeAssistantInput(action=action))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"].endswith("/api/services/light/turn_on")
    assert call["json"] == {"entity_id": "light.living_room"}
    assert call["headers"]["Authorization"] == "Bearer abc"
    assert call["ssl"] is False


@pytest.mark.asyncio
async def test_set_brightness_uses_brightness_pct(monkeypatch):
    cfg = HomeAssistantAPIConfig(
        base_url="http://localhost:8123",
        token="abc",
        devices={"lamp": "light.living_room"},
        verify_ssl=False,
    )
    c = HomeAssistantAPIConnector(cfg)

    fake = _FakeSession()

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout=None: fake)

    action = json.dumps({"device": "lamp", "command": "set", "value": 50})
    await c.connect(HomeAssistantInput(action=action))

    call = fake.calls[0]
    assert call["url"].endswith("/api/services/light/turn_on")
    assert call["json"] == {"entity_id": "light.living_room", "brightness_pct": 50.0}


@pytest.mark.asyncio
async def test_unknown_device_alias_raises(monkeypatch):
    cfg = HomeAssistantAPIConfig(base_url="http://localhost:8123", token="abc", devices={})
    c = HomeAssistantAPIConnector(cfg)

    action = json.dumps({"device": "missing", "command": "on"})
    with pytest.raises(ValueError):
        await c.connect(HomeAssistantInput(action=action))
