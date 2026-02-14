"""
End-to-end simulation test for Home Assistant integration.

Starts a mock Home Assistant HTTP server and tests the full flow:
Provider -> Connector -> HTTP API -> Mock Server
"""

import logging
from typing import Any, Dict, List

import pytest
import pytest_asyncio
from aiohttp import web

from actions.home_assistant.connector.rest_api import (
    HomeAssistantConfig,
    HomeAssistantRESTConnector,
)
from actions.home_assistant.interface import HomeAssistantInput
from inputs.plugins.home_assistant_state import (
    HomeAssistantStateConfig,
    HomeAssistantStateInput,
)
from providers.home_assistant_provider import HomeAssistantProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Mock Home Assistant Server ---


class MockHomeAssistant:
    """
    Mock Home Assistant REST API server.

    Simulates /api/states and /api/services endpoints with
    in-memory device state tracking.
    """

    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {
            "light.living_room": {
                "entity_id": "light.living_room",
                "state": "off",
                "attributes": {
                    "friendly_name": "Living Room Light",
                    "brightness": 0,
                    "supported_features": 1,
                },
            },
            "light.bedroom": {
                "entity_id": "light.bedroom",
                "state": "off",
                "attributes": {
                    "friendly_name": "Bedroom Light",
                    "brightness": 0,
                    "supported_features": 1,
                },
            },
            "climate.thermostat": {
                "entity_id": "climate.thermostat",
                "state": "heat",
                "attributes": {
                    "friendly_name": "Thermostat",
                    "current_temperature": 20.5,
                    "temperature": 21.0,
                    "unit_of_measurement": "°C",
                },
            },
            "sensor.temperature": {
                "entity_id": "sensor.temperature",
                "state": "20.5",
                "attributes": {
                    "friendly_name": "Temperature Sensor",
                    "unit_of_measurement": "°C",
                },
            },
            "switch.garage_door": {
                "entity_id": "switch.garage_door",
                "state": "off",
                "attributes": {
                    "friendly_name": "Garage Door",
                },
            },
            "fan.ceiling": {
                "entity_id": "fan.ceiling",
                "state": "off",
                "attributes": {
                    "friendly_name": "Ceiling Fan",
                    "percentage": 0,
                },
            },
            "lock.front_door": {
                "entity_id": "lock.front_door",
                "state": "locked",
                "attributes": {
                    "friendly_name": "Front Door Lock",
                },
            },
            "cover.blinds": {
                "entity_id": "cover.blinds",
                "state": "closed",
                "attributes": {
                    "friendly_name": "Living Room Blinds",
                    "current_position": 0,
                },
            },
            "media_player.tv": {
                "entity_id": "media_player.tv",
                "state": "off",
                "attributes": {
                    "friendly_name": "Living Room TV",
                    "volume_level": 0.5,
                    "is_volume_muted": False,
                    "source": "HDMI 1",
                },
            },
            "vacuum.roomba": {
                "entity_id": "vacuum.roomba",
                "state": "docked",
                "attributes": {
                    "friendly_name": "Roomba",
                    "battery_level": 100,
                },
            },
            "scene.movie_mode": {
                "entity_id": "scene.movie_mode",
                "state": "scening",
                "attributes": {
                    "friendly_name": "Movie Mode",
                },
            },
            "alarm_control_panel.home": {
                "entity_id": "alarm_control_panel.home",
                "state": "disarmed",
                "attributes": {
                    "friendly_name": "Home Alarm",
                },
            },
        }
        self.service_calls: List[Dict[str, Any]] = []
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/api/states", self._handle_get_states)
        self.app.router.add_get("/api/states/{entity_id}", self._handle_get_state)
        self.app.router.add_post(
            "/api/services/{domain}/{service}", self._handle_call_service
        )

    async def _handle_get_states(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"message": "Unauthorized"}, status=401)
        return web.json_response(list(self.states.values()))

    async def _handle_get_state(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"message": "Unauthorized"}, status=401)

        entity_id = request.match_info["entity_id"]
        if entity_id not in self.states:
            return web.json_response({"message": "Entity not found"}, status=404)
        return web.json_response(self.states[entity_id])

    async def _handle_call_service(self, request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"message": "Unauthorized"}, status=401)

        domain = request.match_info["domain"]
        service = request.match_info["service"]
        data = await request.json()
        entity_id = data.get("entity_id", "")

        self.service_calls.append({"domain": domain, "service": service, "data": data})

        # Apply state changes
        if entity_id in self.states:
            if service == "turn_on":
                self.states[entity_id]["state"] = "on"
                if domain == "light":
                    brightness_pct = data.get("brightness_pct")
                    if brightness_pct is not None:
                        self.states[entity_id]["attributes"]["brightness"] = int(
                            brightness_pct / 100 * 255
                        )
                    else:
                        self.states[entity_id]["attributes"]["brightness"] = 255
                    rgb_color = data.get("rgb_color")
                    if rgb_color is not None:
                        self.states[entity_id]["attributes"]["rgb_color"] = rgb_color
                    color_temp = data.get("color_temp_kelvin")
                    if color_temp is not None:
                        self.states[entity_id]["attributes"][
                            "color_temp_kelvin"
                        ] = color_temp
            elif service == "turn_off":
                self.states[entity_id]["state"] = "off"
                if domain == "light":
                    self.states[entity_id]["attributes"]["brightness"] = 0
            elif service == "toggle":
                current = self.states[entity_id]["state"]
                self.states[entity_id]["state"] = "off" if current == "on" else "on"
            elif service == "set_temperature":
                temp = data.get("temperature")
                if temp is not None:
                    self.states[entity_id]["attributes"]["temperature"] = temp
            elif service == "set_value":
                val = data.get("value")
                if val is not None:
                    self.states[entity_id]["attributes"]["percentage"] = val
                    self.states[entity_id]["state"] = "on" if val > 0 else "off"
            elif service == "set_hvac_mode":
                hvac_mode = data.get("hvac_mode")
                if hvac_mode is not None:
                    self.states[entity_id]["state"] = hvac_mode
            elif service == "lock":
                self.states[entity_id]["state"] = "locked"
            elif service == "unlock":
                self.states[entity_id]["state"] = "unlocked"
            elif service == "open_cover":
                self.states[entity_id]["state"] = "open"
                self.states[entity_id]["attributes"]["current_position"] = 100
            elif service == "close_cover":
                self.states[entity_id]["state"] = "closed"
                self.states[entity_id]["attributes"]["current_position"] = 0
            elif service == "stop_cover":
                self.states[entity_id]["state"] = "stopped"
            elif service == "set_cover_position":
                pos = data.get("position")
                if pos is not None:
                    self.states[entity_id]["attributes"]["current_position"] = pos
                    self.states[entity_id]["state"] = "open" if pos > 0 else "closed"
            elif service == "media_play":
                self.states[entity_id]["state"] = "playing"
            elif service == "media_pause":
                self.states[entity_id]["state"] = "paused"
            elif service == "media_stop":
                self.states[entity_id]["state"] = "idle"
            elif service == "volume_set":
                vol = data.get("volume_level")
                if vol is not None:
                    self.states[entity_id]["attributes"]["volume_level"] = vol
            elif service == "volume_mute":
                muted = data.get("is_volume_muted")
                if muted is not None:
                    self.states[entity_id]["attributes"]["is_volume_muted"] = muted
            elif service == "select_source":
                source = data.get("source")
                if source is not None:
                    self.states[entity_id]["attributes"]["source"] = source
            elif service == "set_percentage":
                pct = data.get("percentage")
                if pct is not None:
                    self.states[entity_id]["attributes"]["percentage"] = pct
                    self.states[entity_id]["state"] = "on" if pct > 0 else "off"
            elif service == "oscillate":
                osc = data.get("oscillating")
                if osc is not None:
                    self.states[entity_id]["attributes"]["oscillating"] = osc
            elif service == "start":
                self.states[entity_id]["state"] = "cleaning"
            elif service == "stop":
                self.states[entity_id]["state"] = "idle"
            elif service == "pause":
                self.states[entity_id]["state"] = "paused"
            elif service == "return_to_base":
                self.states[entity_id]["state"] = "returning"
            elif service in (
                "alarm_arm_home",
                "alarm_arm_away",
                "alarm_arm_night",
            ):
                mode = service.replace("alarm_arm_", "armed_")
                self.states[entity_id]["state"] = mode
            elif service == "alarm_disarm":
                self.states[entity_id]["state"] = "disarmed"

        logger.info(f"[MockHA] {domain}.{service} -> {entity_id} | data={data}")
        return web.json_response([self.states.get(entity_id, {})])


@pytest_asyncio.fixture
async def mock_ha_server():
    """Start mock HA server and return (base_url, mock_ha)."""
    mock_ha = MockHomeAssistant()
    runner = web.AppRunner(mock_ha.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Get the actual bound port
    host, port = site._server.sockets[0].getsockname()  # type: ignore
    base_url = f"http://{host}:{port}"

    yield base_url, mock_ha

    await runner.cleanup()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset provider singleton between tests."""
    HomeAssistantProvider.reset()
    yield
    HomeAssistantProvider.reset()


# --- E2E Tests ---


@pytest.mark.asyncio
async def test_e2e_turn_on_light(mock_ha_server):
    """Simulate: User says 'turn on the living room light'."""
    base_url, mock_ha = mock_ha_server

    # Verify light is initially off
    assert mock_ha.states["light.living_room"]["state"] == "off"

    # Create connector (as OM1 runtime would)
    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-simulation-token",
        devices={
            "living_room_light": "light.living_room",
            "bedroom_light": "light.bedroom",
            "thermostat": "climate.thermostat",
            "garage_door": "switch.garage_door",
        },
    )
    connector = HomeAssistantRESTConnector(config)

    # LLM would produce this output
    llm_output = HomeAssistantInput(device="living_room_light", command="on")
    await connector.connect(llm_output)

    # Verify: light is now on
    assert mock_ha.states["light.living_room"]["state"] == "on"
    assert len(mock_ha.service_calls) == 1
    assert mock_ha.service_calls[0]["service"] == "turn_on"


@pytest.mark.asyncio
async def test_e2e_turn_off_light(mock_ha_server):
    """Simulate: Turn on then turn off a light."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"bedroom_light": "light.bedroom"},
    )
    connector = HomeAssistantRESTConnector(config)

    # Turn on
    await connector.connect(HomeAssistantInput(device="bedroom_light", command="on"))
    assert mock_ha.states["light.bedroom"]["state"] == "on"

    # Turn off
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="bedroom_light", command="off"))
    assert mock_ha.states["light.bedroom"]["state"] == "off"


@pytest.mark.asyncio
async def test_e2e_set_brightness(mock_ha_server):
    """Simulate: User says 'set living room light to 60%'."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"living_room_light": "light.living_room"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(
        device="living_room_light", command="set", value=60.0
    )
    await connector.connect(llm_output)

    # Light should be on with ~60% brightness
    assert mock_ha.states["light.living_room"]["state"] == "on"
    expected_brightness = int(60 / 100 * 255)
    assert (
        mock_ha.states["light.living_room"]["attributes"]["brightness"]
        == expected_brightness
    )

    # Verify the service call payload
    call = mock_ha.service_calls[0]
    assert call["domain"] == "light"
    assert call["service"] == "turn_on"
    assert call["data"]["brightness_pct"] == 60.0


@pytest.mark.asyncio
async def test_e2e_set_thermostat(mock_ha_server):
    """Simulate: User says 'set thermostat to 23 degrees'."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"thermostat": "climate.thermostat"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(device="thermostat", command="set", value=23.0)
    await connector.connect(llm_output)

    assert mock_ha.states["climate.thermostat"]["attributes"]["temperature"] == 23.0

    call = mock_ha.service_calls[0]
    assert call["domain"] == "climate"
    assert call["service"] == "set_temperature"
    assert call["data"]["temperature"] == 23.0


@pytest.mark.asyncio
async def test_e2e_toggle_light(mock_ha_server):
    """Simulate: User says 'toggle the garage door'."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"garage_door": "switch.garage_door"},
    )
    connector = HomeAssistantRESTConnector(config)

    # Toggle: off -> on
    await connector.connect(HomeAssistantInput(device="garage_door", command="toggle"))
    assert mock_ha.states["switch.garage_door"]["state"] == "on"

    # Toggle: on -> off
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="garage_door", command="toggle"))
    assert mock_ha.states["switch.garage_door"]["state"] == "off"


@pytest.mark.asyncio
async def test_e2e_state_polling(mock_ha_server):
    """Simulate: StateInput polls and detects changes."""
    base_url, mock_ha = mock_ha_server

    # Create state input
    state_config = HomeAssistantStateConfig(
        base_url=base_url,
        token="test-token",
        entities=["light.living_room", "sensor.temperature", "climate.thermostat"],
        poll_interval=0.01,
    )
    state_input = HomeAssistantStateInput(state_config)

    # First poll - all states are new, should report all
    raw = await state_input._poll()
    assert raw is not None
    await state_input.raw_to_text(raw)
    assert len(state_input.messages) == 1

    buffer = state_input.formatted_latest_buffer()
    assert buffer is not None
    assert "INPUT: Home Status" in buffer
    assert "Living Room Light" in buffer
    assert "Temperature Sensor" in buffer
    assert "Thermostat" in buffer

    # Second poll - no changes, should not add message
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)
    assert len(state_input.messages) == 0

    # Now change a state on the mock server (simulate light turned on)
    mock_ha.states["light.living_room"]["state"] = "on"
    mock_ha.states["light.living_room"]["attributes"]["brightness"] = 200

    # Third poll - should detect the change
    HomeAssistantProvider.reset()
    state_input.provider = HomeAssistantProvider(base_url=base_url, token="test-token")
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)
    assert len(state_input.messages) == 1

    buffer = state_input.formatted_latest_buffer()
    assert buffer is not None
    assert "Living Room Light: on" in buffer
    assert "brightness" in buffer


@pytest.mark.asyncio
async def test_e2e_full_scenario(mock_ha_server):
    """
    Full scenario simulation:
    1. Poll initial states
    2. User says 'turn on living room light to 80%'
    3. Poll again and detect the change
    """
    base_url, mock_ha = mock_ha_server

    # --- Setup ---
    action_config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={
            "living_room_light": "light.living_room",
            "thermostat": "climate.thermostat",
        },
    )
    connector = HomeAssistantRESTConnector(action_config)

    state_config = HomeAssistantStateConfig(
        base_url=base_url,
        token="test-token",
        entities=["light.living_room", "climate.thermostat"],
        poll_interval=0.01,
    )

    HomeAssistantProvider.reset()
    state_input = HomeAssistantStateInput(state_config)

    # --- Step 1: Initial state poll ---
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)
    initial_buffer = state_input.formatted_latest_buffer()
    assert initial_buffer is not None
    assert "Living Room Light: off" in initial_buffer

    # --- Step 2: User command -> LLM -> Action ---
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(action_config)
    llm_output = HomeAssistantInput(
        device="living_room_light", command="set", value=80.0
    )
    await connector.connect(llm_output)

    # Verify mock server state changed
    assert mock_ha.states["light.living_room"]["state"] == "on"
    assert mock_ha.states["light.living_room"]["attributes"]["brightness"] == int(
        80 / 100 * 255
    )

    # --- Step 3: Next poll detects the change ---
    HomeAssistantProvider.reset()
    state_input.provider = HomeAssistantProvider(base_url=base_url, token="test-token")
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)
    change_buffer = state_input.formatted_latest_buffer()

    assert change_buffer is not None
    assert "Living Room Light: on" in change_buffer
    assert "brightness" in change_buffer


# --- Error handling scenarios ---


@pytest.mark.asyncio
async def test_e2e_unknown_device(mock_ha_server):
    """Simulate: LLM outputs a device alias not in config."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"living_room_light": "light.living_room"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(device="kitchen_light", command="on")
    await connector.connect(llm_output)

    # No service call should have been made
    assert len(mock_ha.service_calls) == 0


@pytest.mark.asyncio
async def test_e2e_unknown_command(mock_ha_server):
    """Simulate: LLM outputs an invalid command."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"living_room_light": "light.living_room"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(device="living_room_light", command="blink")
    await connector.connect(llm_output)

    # No service call should have been made
    assert len(mock_ha.service_calls) == 0
    # State should remain unchanged
    assert mock_ha.states["light.living_room"]["state"] == "off"


@pytest.mark.asyncio
async def test_e2e_set_without_value(mock_ha_server):
    """Simulate: LLM outputs 'set' command without providing a value."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"living_room_light": "light.living_room"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(
        device="living_room_light", command="set", value=None
    )
    await connector.connect(llm_output)

    # No service call should have been made
    assert len(mock_ha.service_calls) == 0
    assert mock_ha.states["light.living_room"]["state"] == "off"


@pytest.mark.asyncio
async def test_e2e_auth_failure(mock_ha_server):
    """Simulate: Provider uses wrong/empty token -> 401 from server."""
    base_url, mock_ha = mock_ha_server

    HomeAssistantProvider.reset()
    provider = HomeAssistantProvider(
        base_url=base_url,
        token="",
        token_env="NONEXISTENT_HA_TOKEN_VAR_99999",
    )

    # get_state should raise due to missing token
    with pytest.raises(ValueError, match="No Home Assistant token found"):
        await provider.get_state("light.living_room")


@pytest.mark.asyncio
async def test_e2e_entity_not_found(mock_ha_server):
    """Simulate: Provider requests a non-existent entity -> 404."""
    base_url, mock_ha = mock_ha_server

    HomeAssistantProvider.reset()
    provider = HomeAssistantProvider(
        base_url=base_url,
        token="test-token",
    )

    with pytest.raises(RuntimeError, match="Failed to get state"):
        await provider.get_state("light.nonexistent_room")


# --- Additional coverage scenarios ---


@pytest.mark.asyncio
async def test_e2e_multiple_state_changes(mock_ha_server):
    """Simulate: Multiple entities change state between polls."""
    base_url, mock_ha = mock_ha_server

    state_config = HomeAssistantStateConfig(
        base_url=base_url,
        token="test-token",
        entities=[
            "light.living_room",
            "light.bedroom",
            "sensor.temperature",
        ],
        poll_interval=0.01,
    )
    state_input = HomeAssistantStateInput(state_config)

    # First poll - initial states
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)
    state_input.formatted_latest_buffer()  # consume

    # Change multiple states at once
    mock_ha.states["light.living_room"]["state"] = "on"
    mock_ha.states["light.living_room"]["attributes"]["brightness"] = 180
    mock_ha.states["light.bedroom"]["state"] = "on"
    mock_ha.states["light.bedroom"]["attributes"]["brightness"] = 100
    mock_ha.states["sensor.temperature"]["state"] = "25.3"

    # Second poll - should detect all 3 changes
    HomeAssistantProvider.reset()
    state_input.provider = HomeAssistantProvider(base_url=base_url, token="test-token")
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)

    buffer = state_input.formatted_latest_buffer()
    assert buffer is not None
    assert "Living Room Light: on" in buffer
    assert "Bedroom Light: on" in buffer
    assert "Temperature Sensor: 25.3" in buffer


@pytest.mark.asyncio
async def test_e2e_set_fan_generic_domain(mock_ha_server):
    """Simulate: Set a fan speed (generic domain, uses set_value service)."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices={"ceiling_fan": "fan.ceiling"},
    )
    connector = HomeAssistantRESTConnector(config)

    llm_output = HomeAssistantInput(device="ceiling_fan", command="set", value=75.0)
    await connector.connect(llm_output)

    # Verify the service call
    assert len(mock_ha.service_calls) == 1
    call = mock_ha.service_calls[0]
    assert call["domain"] == "fan"
    assert call["service"] == "set_value"
    assert call["data"]["value"] == 75.0

    # Verify mock server applied the state
    assert mock_ha.states["fan.ceiling"]["attributes"]["percentage"] == 75.0
    assert mock_ha.states["fan.ceiling"]["state"] == "on"


@pytest.mark.asyncio
async def test_e2e_state_polling_climate_format(mock_ha_server):
    """Simulate: State polling correctly formats climate current_temperature."""
    base_url, mock_ha = mock_ha_server

    state_config = HomeAssistantStateConfig(
        base_url=base_url,
        token="test-token",
        entities=["climate.thermostat"],
        poll_interval=0.01,
    )
    state_input = HomeAssistantStateInput(state_config)

    # First poll
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)

    buffer = state_input.formatted_latest_buffer()
    assert buffer is not None
    assert "Thermostat: heat" in buffer
    assert "current temperature 20.5" in buffer

    # Change thermostat temperature
    mock_ha.states["climate.thermostat"]["state"] = "cool"
    mock_ha.states["climate.thermostat"]["attributes"]["current_temperature"] = 26.0

    # Second poll - detect change
    HomeAssistantProvider.reset()
    state_input.provider = HomeAssistantProvider(base_url=base_url, token="test-token")
    raw = await state_input._poll()
    await state_input.raw_to_text(raw)

    buffer = state_input.formatted_latest_buffer()
    assert buffer is not None
    assert "Thermostat: cool" in buffer
    assert "current temperature 26.0" in buffer


# --- New device type e2e scenarios ---

ALL_DEVICES = {
    "living_room_light": "light.living_room",
    "bedroom_light": "light.bedroom",
    "thermostat": "climate.thermostat",
    "ceiling_fan": "fan.ceiling",
    "front_door": "lock.front_door",
    "blinds": "cover.blinds",
    "tv": "media_player.tv",
    "roomba": "vacuum.roomba",
    "movie_mode": "scene.movie_mode",
    "alarm": "alarm_control_panel.home",
    "garage_door": "switch.garage_door",
}


@pytest.mark.asyncio
async def test_e2e_lock_unlock(mock_ha_server):
    """Simulate: Lock and unlock the front door."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    # Unlock
    await connector.connect(HomeAssistantInput(device="front_door", command="unlock"))
    assert mock_ha.states["lock.front_door"]["state"] == "unlocked"

    # Lock again
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="front_door", command="lock"))
    assert mock_ha.states["lock.front_door"]["state"] == "locked"


@pytest.mark.asyncio
async def test_e2e_cover_open_close_stop(mock_ha_server):
    """Simulate: Open, stop, and close blinds."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    # Open
    await connector.connect(HomeAssistantInput(device="blinds", command="open"))
    assert mock_ha.states["cover.blinds"]["state"] == "open"
    assert mock_ha.states["cover.blinds"]["attributes"]["current_position"] == 100

    # Stop mid-way
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="blinds", command="stop"))
    assert mock_ha.states["cover.blinds"]["state"] == "stopped"

    # Close
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="blinds", command="close"))
    assert mock_ha.states["cover.blinds"]["state"] == "closed"
    assert mock_ha.states["cover.blinds"]["attributes"]["current_position"] == 0


@pytest.mark.asyncio
async def test_e2e_cover_set_position(mock_ha_server):
    """Simulate: Set blinds to 60% open."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(
        HomeAssistantInput(device="blinds", command="set_position", value=60.0)
    )

    assert mock_ha.states["cover.blinds"]["state"] == "open"
    assert mock_ha.states["cover.blinds"]["attributes"]["current_position"] == 60.0


@pytest.mark.asyncio
async def test_e2e_media_player_full_flow(mock_ha_server):
    """Simulate: Turn on TV, play, set volume, select source, mute, stop."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )

    # Turn on
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="on"))
    assert mock_ha.states["media_player.tv"]["state"] == "on"

    # Play
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="play"))
    assert mock_ha.states["media_player.tv"]["state"] == "playing"

    # Volume set to 30%
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(
        HomeAssistantInput(device="tv", command="volume_set", value=30.0)
    )
    assert mock_ha.states["media_player.tv"]["attributes"]["volume_level"] == 0.3

    # Select source
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(
        HomeAssistantInput(device="tv", command="select_source", mode="Netflix")
    )
    assert mock_ha.states["media_player.tv"]["attributes"]["source"] == "Netflix"

    # Mute
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="volume_mute"))
    assert mock_ha.states["media_player.tv"]["attributes"]["is_volume_muted"] is True

    # Unmute
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="volume_unmute"))
    assert mock_ha.states["media_player.tv"]["attributes"]["is_volume_muted"] is False

    # Pause
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="pause"))
    assert mock_ha.states["media_player.tv"]["state"] == "paused"

    # Stop
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="tv", command="media_stop"))
    assert mock_ha.states["media_player.tv"]["state"] == "idle"


@pytest.mark.asyncio
async def test_e2e_vacuum_full_flow(mock_ha_server):
    """Simulate: Start vacuum, pause, resume, return to base."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )

    # Start cleaning
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="roomba", command="start"))
    assert mock_ha.states["vacuum.roomba"]["state"] == "cleaning"

    # Pause
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="roomba", command="vacuum_pause"))
    assert mock_ha.states["vacuum.roomba"]["state"] == "paused"

    # Stop
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="roomba", command="stop"))
    assert mock_ha.states["vacuum.roomba"]["state"] == "idle"

    # Return to base
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(
        HomeAssistantInput(device="roomba", command="return_to_base")
    )
    assert mock_ha.states["vacuum.roomba"]["state"] == "returning"


@pytest.mark.asyncio
async def test_e2e_alarm_full_flow(mock_ha_server):
    """Simulate: Arm home, arm away, arm night, disarm."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )

    assert mock_ha.states["alarm_control_panel.home"]["state"] == "disarmed"

    # Arm home
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="alarm", command="arm_home"))
    assert mock_ha.states["alarm_control_panel.home"]["state"] == "armed_home"

    # Arm away
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="alarm", command="arm_away"))
    assert mock_ha.states["alarm_control_panel.home"]["state"] == "armed_away"

    # Arm night
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="alarm", command="arm_night"))
    assert mock_ha.states["alarm_control_panel.home"]["state"] == "armed_night"

    # Disarm
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(HomeAssistantInput(device="alarm", command="disarm"))
    assert mock_ha.states["alarm_control_panel.home"]["state"] == "disarmed"


@pytest.mark.asyncio
async def test_e2e_scene_activate(mock_ha_server):
    """Simulate: Activate movie mode scene."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(HomeAssistantInput(device="movie_mode", command="activate"))

    assert len(mock_ha.service_calls) == 1
    call = mock_ha.service_calls[0]
    assert call["domain"] == "scene"
    assert call["service"] == "turn_on"


@pytest.mark.asyncio
async def test_e2e_light_set_color(mock_ha_server):
    """Simulate: Set living room light to red."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(
        HomeAssistantInput(
            device="living_room_light", command="set_color", mode="#FF0000"
        )
    )

    assert mock_ha.states["light.living_room"]["state"] == "on"
    assert mock_ha.states["light.living_room"]["attributes"]["rgb_color"] == [
        255,
        0,
        0,
    ]


@pytest.mark.asyncio
async def test_e2e_light_set_color_temp(mock_ha_server):
    """Simulate: Set living room light to warm white (3000K)."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(
        HomeAssistantInput(
            device="living_room_light", command="set_color_temp", value=3000.0
        )
    )

    assert mock_ha.states["light.living_room"]["state"] == "on"
    assert (
        mock_ha.states["light.living_room"]["attributes"]["color_temp_kelvin"] == 3000.0
    )


@pytest.mark.asyncio
async def test_e2e_climate_set_hvac_mode(mock_ha_server):
    """Simulate: Set thermostat to cool mode."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(
        HomeAssistantInput(device="thermostat", command="set_hvac_mode", mode="cool")
    )

    assert mock_ha.states["climate.thermostat"]["state"] == "cool"


@pytest.mark.asyncio
async def test_e2e_fan_set_percentage(mock_ha_server):
    """Simulate: Set ceiling fan to 50% speed."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )
    connector = HomeAssistantRESTConnector(config)

    await connector.connect(
        HomeAssistantInput(device="ceiling_fan", command="set_percentage", value=50.0)
    )

    assert mock_ha.states["fan.ceiling"]["state"] == "on"
    assert mock_ha.states["fan.ceiling"]["attributes"]["percentage"] == 50.0


@pytest.mark.asyncio
async def test_e2e_fan_oscillate(mock_ha_server):
    """Simulate: Turn on and off oscillation on ceiling fan."""
    base_url, mock_ha = mock_ha_server

    config = HomeAssistantConfig(
        base_url=base_url,
        token="test-token",
        devices=ALL_DEVICES,
    )

    # Start oscillation
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(
        HomeAssistantInput(device="ceiling_fan", command="oscillate")
    )
    assert mock_ha.states["fan.ceiling"]["attributes"]["oscillating"] is True

    # Stop oscillation
    HomeAssistantProvider.reset()
    connector = HomeAssistantRESTConnector(config)
    await connector.connect(
        HomeAssistantInput(device="ceiling_fan", command="stop_oscillate")
    )
    assert mock_ha.states["fan.ceiling"]["attributes"]["oscillating"] is False
