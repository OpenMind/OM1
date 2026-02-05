"""
Tests for Home Assistant Action Connector.

This module contains unit tests for the Home Assistant REST API connector,
testing light control, switch control, and thermostat control functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from actions.home_assistant.interface import (
    HomeAssistantInput,
    HomeAssistantOutput,
    DeviceType,
    LightAction,
    SwitchAction,
    ThermostatAction,
)
from actions.home_assistant.connector.rest_api import (
    HomeAssistantConfig,
    HomeAssistantRESTConnector,
)


@pytest.fixture
def connector_config():
    """Create a test configuration."""
    return HomeAssistantConfig(
        base_url="http://localhost:8123",
        access_token="test_token_12345",
        verify_ssl=False,
        timeout=5,
    )


@pytest.fixture
def connector(connector_config):
    """Create a test connector instance."""
    return HomeAssistantRESTConnector(connector_config)


class TestHomeAssistantConfig:
    """Tests for HomeAssistantConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HomeAssistantConfig()
        assert config.base_url == "http://homeassistant.local:8123"
        assert config.access_token == ""
        assert config.verify_ssl is True
        assert config.timeout == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HomeAssistantConfig(
            base_url="http://192.168.1.100:8123",
            access_token="my_token",
            verify_ssl=False,
            timeout=30,
        )
        assert config.base_url == "http://192.168.1.100:8123"
        assert config.access_token == "my_token"
        assert config.verify_ssl is False
        assert config.timeout == 30


class TestHomeAssistantRESTConnector:
    """Tests for HomeAssistantRESTConnector."""

    def test_initialization(self, connector, connector_config):
        """Test connector initialization."""
        assert connector.base_url == "http://localhost:8123"
        assert connector.access_token == "test_token_12345"
        assert connector.verify_ssl is False
        assert connector.timeout == 5

    def test_get_headers(self, connector):
        """Test HTTP headers generation."""
        headers = connector._get_headers()
        assert headers["Authorization"] == "Bearer test_token_12345"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_control_light_turn_on(self, connector):
        """Test turning on a light."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "on"}

                result = await connector._control_light(
                    entity_id="light.living_room",
                    action="turn_on",
                )

                assert result.success is True
                assert "light.living_room" in result.message
                mock_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_control_light_with_brightness(self, connector):
        """Test setting light brightness."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "on"}

                result = await connector._control_light(
                    entity_id="light.bedroom",
                    action="brightness",
                    brightness=128,
                )

                assert result.success is True
                call_args = mock_service.call_args
                assert call_args[0][0] == "light"
                assert call_args[0][1] == "turn_on"
                assert call_args[0][2]["brightness"] == 128

    @pytest.mark.asyncio
    async def test_control_light_with_color(self, connector):
        """Test setting light color."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "on"}

                result = await connector._control_light(
                    entity_id="light.bedroom",
                    action="color",
                    color_rgb="255,0,0",
                )

                assert result.success is True
                call_args = mock_service.call_args
                assert call_args[0][2]["rgb_color"] == [255, 0, 0]

    @pytest.mark.asyncio
    async def test_control_light_invalid_color(self, connector):
        """Test invalid color format handling."""
        result = await connector._control_light(
            entity_id="light.bedroom",
            action="color",
            color_rgb="invalid",
        )

        assert result.success is False
        assert "Invalid RGB color format" in result.message

    @pytest.mark.asyncio
    async def test_control_switch_turn_on(self, connector):
        """Test turning on a switch."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "on"}

                result = await connector._control_switch(
                    entity_id="switch.coffee_maker",
                    action="turn_on",
                )

                assert result.success is True
                mock_service.assert_called_once_with(
                    "switch", "turn_on", {"entity_id": "switch.coffee_maker"}
                )

    @pytest.mark.asyncio
    async def test_control_switch_toggle(self, connector):
        """Test toggling a switch."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "off"}

                result = await connector._control_switch(
                    entity_id="switch.fan",
                    action="toggle",
                )

                assert result.success is True
                mock_service.assert_called_once_with(
                    "switch", "toggle", {"entity_id": "switch.fan"}
                )

    @pytest.mark.asyncio
    async def test_control_thermostat_set_temperature(self, connector):
        """Test setting thermostat temperature."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "heat"}

                result = await connector._control_thermostat(
                    entity_id="climate.living_room",
                    action="set_temperature",
                    temperature=22.5,
                )

                assert result.success is True
                call_args = mock_service.call_args
                assert call_args[0][0] == "climate"
                assert call_args[0][1] == "set_temperature"
                assert call_args[0][2]["temperature"] == 22.5

    @pytest.mark.asyncio
    async def test_control_thermostat_set_hvac_mode(self, connector):
        """Test setting thermostat HVAC mode."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "cool"}

                result = await connector._control_thermostat(
                    entity_id="climate.bedroom",
                    action="set_hvac_mode",
                    hvac_mode="cool",
                )

                assert result.success is True
                call_args = mock_service.call_args
                assert call_args[0][2]["hvac_mode"] == "cool"

    @pytest.mark.asyncio
    async def test_control_cover_open(self, connector):
        """Test opening a cover."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "open"}

                result = await connector._control_cover(
                    entity_id="cover.garage",
                    action="open",
                )

                assert result.success is True
                mock_service.assert_called_once_with(
                    "cover", "open_cover", {"entity_id": "cover.garage"}
                )

    @pytest.mark.asyncio
    async def test_control_fan_turn_on(self, connector):
        """Test turning on a fan."""
        with patch.object(
            connector, "_call_service", new_callable=AsyncMock
        ) as mock_service:
            mock_service.return_value = {"success": True, "data": []}
            with patch.object(
                connector, "_get_entity_state", new_callable=AsyncMock
            ) as mock_state:
                mock_state.return_value = {"state": "on"}

                result = await connector._control_fan(
                    entity_id="fan.bedroom",
                    action="turn_on",
                )

                assert result.success is True
                mock_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_routes_to_light(self, connector):
        """Test connect method routes light devices correctly."""
        with patch.object(
            connector, "_control_light", new_callable=AsyncMock
        ) as mock_light:
            mock_light.return_value = HomeAssistantOutput(
                success=True,
                message="Success",
                entity_id="light.test",
            )

            input_data = HomeAssistantInput(
                device_type="light",
                entity_id="light.test",
                action="turn_on",
                brightness=200,
            )

            await connector.connect(input_data)

            mock_light.assert_called_once_with(
                entity_id="light.test",
                action="turn_on",
                brightness=200,
                color_rgb=None,
            )

    @pytest.mark.asyncio
    async def test_connect_unsupported_device_type(self, connector):
        """Test handling of unsupported device types."""
        input_data = HomeAssistantInput(
            device_type="unsupported",
            entity_id="unsupported.test",
            action="test",
        )

        await connector.connect(input_data)

        assert connector._last_result.success is False
        assert "Unsupported device type" in connector._last_result.message


class TestHomeAssistantInterface:
    """Tests for interface classes."""

    def test_home_assistant_input_creation(self):
        """Test HomeAssistantInput dataclass creation."""
        input_data = HomeAssistantInput(
            device_type="light",
            entity_id="light.living_room",
            action="turn_on",
            brightness=255,
            color_rgb="255,255,0",
        )

        assert input_data.device_type == "light"
        assert input_data.entity_id == "light.living_room"
        assert input_data.action == "turn_on"
        assert input_data.brightness == 255
        assert input_data.color_rgb == "255,255,0"

    def test_home_assistant_input_defaults(self):
        """Test HomeAssistantInput default values."""
        input_data = HomeAssistantInput(
            device_type="switch",
            entity_id="switch.test",
            action="toggle",
        )

        assert input_data.brightness is None
        assert input_data.color_rgb is None
        assert input_data.temperature is None
        assert input_data.hvac_mode is None

    def test_home_assistant_output_creation(self):
        """Test HomeAssistantOutput dataclass creation."""
        output = HomeAssistantOutput(
            success=True,
            message="Light turned on successfully",
            entity_id="light.living_room",
            new_state="on",
        )

        assert output.success is True
        assert "successfully" in output.message
        assert output.entity_id == "light.living_room"
        assert output.new_state == "on"

    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        assert DeviceType.LIGHT.value == "light"
        assert DeviceType.SWITCH.value == "switch"
        assert DeviceType.THERMOSTAT.value == "climate"
        assert DeviceType.COVER.value == "cover"
        assert DeviceType.FAN.value == "fan"

    def test_light_action_enum(self):
        """Test LightAction enum values."""
        assert LightAction.ON.value == "turn_on"
        assert LightAction.OFF.value == "turn_off"
        assert LightAction.TOGGLE.value == "toggle"
        assert LightAction.BRIGHTNESS.value == "brightness"
        assert LightAction.COLOR.value == "color"

    def test_thermostat_action_enum(self):
        """Test ThermostatAction enum values."""
        assert ThermostatAction.SET_TEMPERATURE.value == "set_temperature"
        assert ThermostatAction.SET_HVAC_MODE.value == "set_hvac_mode"
        assert ThermostatAction.SET_FAN_MODE.value == "set_fan_mode"
