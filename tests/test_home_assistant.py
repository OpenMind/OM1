"""
Tests for Home Assistant Action Module

Run with: uv run pytest tests/test_home_assistant.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from actions.home_assistant.interface import (
    HomeAssistant,
    HomeAssistantInput,
    HomeAssistantOutput,
)
from actions.home_assistant.connector.home_assistant_api import (
    HomeAssistantAPIConfig,
    HomeAssistantAPIConnector,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return HomeAssistantAPIConfig(
        base_url="http://localhost:8123",
        access_token="test_token_12345",
        verify_ssl=False,
        timeout=10
    )


@pytest.fixture
def connector(config):
    """Create test connector."""
    return HomeAssistantAPIConnector(config)


class TestHomeAssistantInterface:
    """Tests for interface classes."""

    def test_input_creation(self):
        """Test HomeAssistantInput creation."""
        input_data = HomeAssistantInput(
            action="turn_on",
            entity_id="light.living_room"
        )
        assert input_data.action == "turn_on"
        assert input_data.entity_id == "light.living_room"
        assert input_data.brightness is None

    def test_input_with_brightness(self):
        """Test HomeAssistantInput with brightness."""
        input_data = HomeAssistantInput(
            action="set_brightness",
            entity_id="light.bedroom",
            brightness=128
        )
        assert input_data.brightness == 128

    def test_input_with_rgb_color(self):
        """Test HomeAssistantInput with RGB color."""
        input_data = HomeAssistantInput(
            action="set_color",
            entity_id="light.desk",
            rgb_color=(255, 0, 0)
        )
        assert input_data.rgb_color == (255, 0, 0)

    def test_input_with_temperature(self):
        """Test HomeAssistantInput with temperature."""
        input_data = HomeAssistantInput(
            action="set_temperature",
            entity_id="climate.thermostat",
            temperature=22.5
        )
        assert input_data.temperature == 22.5

    def test_output_creation(self):
        """Test HomeAssistantOutput creation."""
        output = HomeAssistantOutput(
            success=True,
            state="on",
            message="Light turned on successfully"
        )
        assert output.success is True
        assert output.state == "on"


class TestHomeAssistantAPIConfig:
    """Tests for API configuration."""

    def test_config_creation(self, config):
        """Test config creation with all fields."""
        assert config.base_url == "http://localhost:8123"
        assert config.access_token == "test_token_12345"
        assert config.verify_ssl is False
        assert config.timeout == 10

    def test_config_defaults(self):
        """Test config with default values."""
        config = HomeAssistantAPIConfig(
            base_url="http://localhost:8123",
            access_token="token"
        )
        assert config.verify_ssl is True
        assert config.timeout == 10


class TestHomeAssistantAPIConnector:
    """Tests for API connector."""

    def test_connector_initialization(self, connector, config):
        """Test connector initialization."""
        assert connector.config == config
        assert "Authorization" in connector._headers
        assert connector._headers["Authorization"] == f"Bearer {config.access_token}"

    def test_get_api_url(self, connector):
        """Test URL construction."""
        url = connector._get_api_url("states/light.test")
        assert url == "http://localhost:8123/api/states/light.test"

    def test_get_api_url_with_trailing_slash(self, config):
        """Test URL construction with trailing slash."""
        config.base_url = "http://localhost:8123/"
        connector = HomeAssistantAPIConnector(config)
        url = connector._get_api_url("services/light/turn_on")
        assert url == "http://localhost:8123/api/services/light/turn_on"

    def test_get_domain_from_entity(self, connector):
        """Test domain extraction from entity_id."""
        assert connector._get_domain_from_entity("light.living_room") == "light"
        assert connector._get_domain_from_entity("switch.fan") == "switch"
        assert connector._get_domain_from_entity("climate.thermostat") == "climate"
        assert connector._get_domain_from_entity("invalid") == ""


class TestHomeAssistantActions:
    """Tests for action execution."""

    @pytest.mark.asyncio
    async def test_turn_on_light(self, connector):
        """Test turning on a light."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="turn_on",
                entity_id="light.living_room"
            )
            result = await connector.connect(input_data)

            assert result.success is True
            assert result.state == "on"
            mock_service.assert_called_once_with(
                "light", "turn_on", {"entity_id": "light.living_room"}
            )

    @pytest.mark.asyncio
    async def test_turn_off_switch(self, connector):
        """Test turning off a switch."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="turn_off",
                entity_id="switch.fan"
            )
            result = await connector.connect(input_data)

            assert result.success is True
            assert result.state == "off"

    @pytest.mark.asyncio
    async def test_set_brightness(self, connector):
        """Test setting light brightness."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="set_brightness",
                entity_id="light.bedroom",
                brightness=128
            )
            result = await connector.connect(input_data)

            assert result.success is True
            assert "brightness=128" in result.state
            mock_service.assert_called_once_with(
                "light", "turn_on", {"entity_id": "light.bedroom", "brightness": 128}
            )

    @pytest.mark.asyncio
    async def test_set_brightness_clamp(self, connector):
        """Test brightness value clamping."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            # Test brightness > 255
            input_data = HomeAssistantInput(
                action="set_brightness",
                entity_id="light.test",
                brightness=300
            )
            await connector.connect(input_data)
            call_args = mock_service.call_args[0][2]
            assert call_args["brightness"] == 255

    @pytest.mark.asyncio
    async def test_set_color(self, connector):
        """Test setting light color."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="set_color",
                entity_id="light.desk",
                rgb_color=(255, 128, 0)
            )
            result = await connector.connect(input_data)

            assert result.success is True
            mock_service.assert_called_once_with(
                "light", "turn_on",
                {"entity_id": "light.desk", "rgb_color": [255, 128, 0]}
            )

    @pytest.mark.asyncio
    async def test_set_temperature(self, connector):
        """Test setting thermostat temperature."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="set_temperature",
                entity_id="climate.living_room",
                temperature=22.0
            )
            result = await connector.connect(input_data)

            assert result.success is True
            mock_service.assert_called_once_with(
                "climate", "set_temperature",
                {"entity_id": "climate.living_room", "temperature": 22.0}
            )

    @pytest.mark.asyncio
    async def test_get_state(self, connector):
        """Test getting device state."""
        with patch.object(connector, 'get_state', new_callable=AsyncMock) as mock_state:
            mock_state.return_value = {"state": "on", "entity_id": "light.test"}

            input_data = HomeAssistantInput(
                action="get_state",
                entity_id="light.test"
            )
            result = await connector.connect(input_data)

            assert result.success is True
            assert result.state == "on"

    @pytest.mark.asyncio
    async def test_toggle_action(self, connector):
        """Test toggle action."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {}

            input_data = HomeAssistantInput(
                action="toggle",
                entity_id="switch.desk_lamp"
            )
            result = await connector.connect(input_data)

            assert result.success is True
            assert result.state == "toggled"

    @pytest.mark.asyncio
    async def test_unknown_action(self, connector):
        """Test unknown action returns error."""
        input_data = HomeAssistantInput(
            action="invalid_action",
            entity_id="light.test"
        )
        result = await connector.connect(input_data)

        assert result.success is False
        assert "Unknown action" in result.message

    @pytest.mark.asyncio
    async def test_missing_entity_id(self, connector):
        """Test missing entity_id returns error."""
        input_data = HomeAssistantInput(
            action="turn_on",
            entity_id=""
        )
        result = await connector.connect(input_data)

        assert result.success is False
        assert "Entity ID is required" in result.message

    @pytest.mark.asyncio
    async def test_brightness_on_non_light(self, connector):
        """Test set_brightness on non-light entity returns error."""
        input_data = HomeAssistantInput(
            action="set_brightness",
            entity_id="switch.fan",
            brightness=128
        )
        result = await connector.connect(input_data)

        assert result.success is False
        assert "light entities" in result.message

    @pytest.mark.asyncio
    async def test_temperature_on_non_climate(self, connector):
        """Test set_temperature on non-climate entity returns error."""
        input_data = HomeAssistantInput(
            action="set_temperature",
            entity_id="light.test",
            temperature=22.0
        )
        result = await connector.connect(input_data)

        assert result.success is False
        assert "climate entities" in result.message

    @pytest.mark.asyncio
    async def test_api_error_handling(self, connector):
        """Test API error handling."""
        with patch.object(connector, 'call_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {"error": "Connection refused", "status": 500}

            input_data = HomeAssistantInput(
                action="turn_on",
                entity_id="light.test"
            )
            result = await connector.connect(input_data)

            assert result.success is False
            assert "Failed" in result.message
