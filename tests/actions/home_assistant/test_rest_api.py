from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.home_assistant.connector.rest_api import (
    HomeAssistantConfig,
    HomeAssistantConnector,
    extract_temperature,
)
from actions.home_assistant.interface import HomeAssistantInput


class TestExtractTemperature:
    """Tests for temperature extraction from text."""

    def test_extract_simple_number(self):
        assert extract_temperature("set temperature 24") == 24.0

    def test_extract_with_degrees(self):
        assert extract_temperature("24 degrees") == 24.0

    def test_extract_with_celsius(self):
        assert extract_temperature("set to 22 celsius") == 22.0

    def test_extract_decimal(self):
        assert extract_temperature("25.5 degrees") == 25.5

    def test_out_of_range_high(self):
        assert extract_temperature("set to 35 degrees") is None

    def test_out_of_range_low(self):
        assert extract_temperature("set to 10 degrees") is None

    def test_no_temperature(self):
        assert extract_temperature("turn on the fan") is None


class TestHomeAssistantConfig:
    """Tests for HomeAssistantConfig."""

    def test_default_values(self):
        config = HomeAssistantConfig()
        assert config.ha_url == "http://homeassistant.local:8123"
        assert config.access_token is None
        assert config.switch_entity_id is None
        assert config.climate_entity_id is None
        assert config.light_entity_id is None

    def test_custom_values(self):
        config = HomeAssistantConfig(
            ha_url="http://192.168.0.100:8123",
            access_token="test_token",
            switch_entity_id="switch.tapo_plug",
            climate_entity_id="climate.lg_ac",
            light_entity_id="light.living_room",
        )
        assert config.ha_url == "http://192.168.0.100:8123"
        assert config.access_token == "test_token"
        assert config.switch_entity_id == "switch.tapo_plug"
        assert config.climate_entity_id == "climate.lg_ac"
        assert config.light_entity_id == "light.living_room"


class TestHomeAssistantConnector:
    """Tests for HomeAssistantConnector."""

    @pytest.fixture
    def config(self):
        return HomeAssistantConfig(
            ha_url="http://homeassistant.local:8123",
            access_token="test_token",
            switch_entity_id="switch.tapo_plug",
            climate_entity_id="climate.lg_ac",
            light_entity_id="light.living_room",
        )

    @pytest.fixture
    def connector(self, config):
        with patch("actions.home_assistant.connector.rest_api.IOProvider"):
            conn = HomeAssistantConnector(config)
            HomeAssistantConnector._last_action = None
            return conn

    def test_init(self, connector):
        assert connector.ha_url == "http://homeassistant.local:8123"
        assert connector.access_token == "test_token"
        assert connector.switch_entity_id == "switch.tapo_plug"

    def test_get_headers(self, connector):
        headers = connector._get_headers()
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Content-Type"] == "application/json"

    def test_parse_action_switch_on(self, connector):
        commands = connector._parse_action("turn on switch")
        assert len(commands) == 1
        assert commands[0]["domain"] == "switch"
        assert commands[0]["service"] == "turn_on"
        assert commands[0]["entity_id"] == "switch.tapo_plug"

    def test_parse_action_switch_off(self, connector):
        commands = connector._parse_action("turn off plug")
        assert len(commands) == 1
        assert commands[0]["domain"] == "switch"
        assert commands[0]["service"] == "turn_off"

    def test_parse_action_fan_on(self, connector):
        commands = connector._parse_action("turn on fan")
        assert len(commands) == 1
        assert commands[0]["domain"] == "switch"
        assert commands[0]["service"] == "turn_on"

    def test_parse_action_light_on(self, connector):
        commands = connector._parse_action("turn on light")
        assert len(commands) == 1
        assert commands[0]["domain"] == "light"
        assert commands[0]["service"] == "turn_on"
        assert commands[0]["entity_id"] == "light.living_room"

    def test_parse_action_light_off(self, connector):
        commands = connector._parse_action("turn off lamp")
        assert len(commands) == 1
        assert commands[0]["domain"] == "light"
        assert commands[0]["service"] == "turn_off"

    def test_parse_action_temperature(self, connector):
        commands = connector._parse_action("set temperature to 24 degrees")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_temperature"
        assert commands[0]["data"]["temperature"] == 24.0

    def test_parse_action_ac_on(self, connector):
        commands = connector._parse_action("turn on ac")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_temperature"
        assert commands[0]["data"]["temperature"] == 24

    def test_parse_action_ac_off(self, connector):
        commands = connector._parse_action("turn off ac")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "off"

    def test_parse_action_cool_mode(self, connector):
        commands = connector._parse_action("set to cool mode")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "cool"

    def test_parse_action_heat_mode(self, connector):
        commands = connector._parse_action("set to heat mode")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "heat"

    def test_parse_action_dry_mode(self, connector):
        commands = connector._parse_action("set to dry mode")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "dry"

    def test_parse_action_auto_mode(self, connector):
        commands = connector._parse_action("set to auto mode")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "auto"

    def test_parse_action_fan_only_mode(self, connector):
        commands = connector._parse_action("set to fan mode")
        assert len(commands) == 1
        assert commands[0]["domain"] == "climate"
        assert commands[0]["service"] == "set_hvac_mode"
        assert commands[0]["data"]["hvac_mode"] == "fan_only"

    def test_parse_action_idle(self, connector):
        commands = connector._parse_action("idle")
        assert len(commands) == 0

    def test_parse_action_no_match(self, connector):
        commands = connector._parse_action("do something random")
        assert len(commands) == 0

    @pytest.mark.asyncio
    async def test_connect_idle(self, connector):
        input_data = HomeAssistantInput(action="idle")
        await connector.connect(input_data)
        # Should not raise, just log idle

    @pytest.mark.asyncio
    async def test_connect_skip_duplicate(self, connector):
        HomeAssistantConnector._last_action = "turn on switch"
        input_data = HomeAssistantInput(action="turn on switch")
        await connector.connect(input_data)
        # Should skip duplicate action

    @pytest.mark.asyncio
    async def test_call_service_success(self, connector):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        connector._session = mock_session

        result = await connector._call_service(
            domain="switch", service="turn_on", entity_id="switch.tapo_plug"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_call_service_error(self, connector):
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        connector._session = mock_session

        result = await connector._call_service(
            domain="switch", service="turn_on", entity_id="switch.tapo_plug"
        )

        assert result is False


class TestHomeAssistantInput:
    """Tests for HomeAssistantInput dataclass."""

    def test_create_input(self):
        input_data = HomeAssistantInput(action="turn on switch")
        assert input_data.action == "turn on switch"

    def test_empty_action(self):
        input_data = HomeAssistantInput(action="")
        assert input_data.action == ""
