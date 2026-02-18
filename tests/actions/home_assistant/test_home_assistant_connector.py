"""Tests for Home Assistant connector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from actions.home_assistant.connector.home_assistant_api import (
    HomeAssistantConfig,
    HomeAssistantConnector,
)
from actions.home_assistant.interface import (
    HomeAssistantAction,
    HomeAssistantDeviceType,
    HomeAssistantInput,
)


class TestHomeAssistantConnector:
    """Test suite for HomeAssistantConnector."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return HomeAssistantConfig(
            base_url="http://test.local:8123",
            token="test_token_12345",
        )

    @pytest.fixture
    def connector(self, config):
        """Create a test connector."""
        return HomeAssistantConnector(config)

    @pytest.mark.asyncio
    async def test_connect_turn_on_light(self, connector):
        """Test turning on a light."""
        input_data = HomeAssistantInput(
            device_type=HomeAssistantDeviceType.LIGHT,
            device_id="light.living_room",
            action=HomeAssistantAction.TURN_ON,
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="{}")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_connect_turn_off_switch(self, connector):
        """Test turning off a switch."""
        input_data = HomeAssistantInput(
            device_type=HomeAssistantDeviceType.SWITCH,
            device_id="switch.kitchen",
            action=HomeAssistantAction.TURN_OFF,
        )

        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_connect_set_brightness(self, connector):
        """Test setting light brightness."""
        input_data = HomeAssistantInput(
            device_type=HomeAssistantDeviceType.LIGHT,
            device_id="light.bedroom",
            action=HomeAssistantAction.SET_BRIGHTNESS,
            brightness=128,
        )

        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_connect_set_temperature(self, connector):
        """Test setting thermostat temperature."""
        input_data = HomeAssistantInput(
            device_type=HomeAssistantDeviceType.THERMOSTAT,
            device_id="climate.living_room",
            action=HomeAssistantAction.SET_TEMPERATURE,
            temperature=22.5,
        )

        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.closed = False

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_stop(self, connector):
        """Test stopping the connector."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        connector._session = mock_session

        await connector.stop()
        mock_session.close.assert_called_once()

    def test_get_headers(self, connector):
        """Test header generation."""
        headers = connector._get_headers()
        assert headers["Authorization"] == "Bearer test_token_12345"
        assert headers["Content-Type"] == "application/json"

    def test_config_defaults(self):
        """Test default config values."""
        config = HomeAssistantConfig()
        assert config.base_url == "http://homeassistant.local:8123"
        assert config.token == ""
