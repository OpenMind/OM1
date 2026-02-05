"""
Tests for Home Assistant State Input Plugin.

This module contains unit tests for the Home Assistant state monitoring
input plugin, testing device state polling and formatting.
"""

import pytest
import time
from unittest.mock import AsyncMock, patch

import aiohttp

from inputs.plugins.home_assistant_state import (
    HomeAssistantInputConfig,
    HomeAssistantStateInput,
    DeviceState,
)
from inputs.base import Message


@pytest.fixture
def input_config():
    """Create a test configuration."""
    return HomeAssistantInputConfig(
        base_url="http://localhost:8123",
        access_token="test_token_12345",
        entity_ids=[
            "light.living_room",
            "switch.coffee_maker",
            "climate.bedroom",
        ],
        poll_interval=1.0,
        verify_ssl=False,
        report_all_states=True,
        input_name="Test Smart Home",
    )


@pytest.fixture
def state_input(input_config):
    """Create a test input instance."""
    return HomeAssistantStateInput(input_config)


class TestHomeAssistantInputConfig:
    """Tests for HomeAssistantInputConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HomeAssistantInputConfig()
        assert config.base_url == "http://homeassistant.local:8123"
        assert config.access_token == ""
        assert config.entity_ids == []
        assert config.poll_interval == 5.0
        assert config.verify_ssl is True
        assert config.report_all_states is False
        assert config.input_name == "Smart Home Devices"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HomeAssistantInputConfig(
            base_url="http://192.168.1.100:8123",
            access_token="my_token",
            entity_ids=["light.test", "switch.test"],
            poll_interval=10.0,
            verify_ssl=False,
            report_all_states=True,
            input_name="My Smart Home",
        )
        assert config.base_url == "http://192.168.1.100:8123"
        assert config.access_token == "my_token"
        assert len(config.entity_ids) == 2
        assert config.poll_interval == 10.0


class TestDeviceState:
    """Tests for DeviceState dataclass."""

    def test_device_state_creation(self):
        """Test DeviceState creation."""
        state = DeviceState(
            entity_id="light.living_room",
            state="on",
            friendly_name="Living Room Light",
            attributes={"brightness": 255},
            last_changed="2025-02-05T10:00:00Z",
        )

        assert state.entity_id == "light.living_room"
        assert state.state == "on"
        assert state.friendly_name == "Living Room Light"
        assert state.attributes["brightness"] == 255


class TestHomeAssistantStateInput:
    """Tests for HomeAssistantStateInput."""

    def test_initialization(self, state_input, input_config):
        """Test input initialization."""
        assert state_input.base_url == "http://localhost:8123"
        assert state_input.access_token == "test_token_12345"
        assert len(state_input.entity_ids) == 3
        assert state_input.poll_interval == 1.0
        assert state_input.descriptor_for_LLM == "Test Smart Home"

    def test_get_headers(self, state_input):
        """Test HTTP headers generation."""
        headers = state_input._get_headers()
        assert headers["Authorization"] == "Bearer test_token_12345"
        assert headers["Content-Type"] == "application/json"

    def test_format_light_state_on(self, state_input):
        """Test formatting light state when on."""
        state = DeviceState(
            entity_id="light.living_room",
            state="on",
            friendly_name="Living Room Light",
            attributes={"brightness": 255},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Living Room Light" in formatted
        assert "ON" in formatted
        assert "100%" in formatted

    def test_format_light_state_with_color(self, state_input):
        """Test formatting light state with color."""
        state = DeviceState(
            entity_id="light.bedroom",
            state="on",
            friendly_name="Bedroom Light",
            attributes={"brightness": 128, "rgb_color": [255, 0, 0]},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Bedroom Light" in formatted
        assert "RGB" in formatted
        assert "50%" in formatted

    def test_format_light_state_off(self, state_input):
        """Test formatting light state when off."""
        state = DeviceState(
            entity_id="light.test",
            state="off",
            friendly_name="Test Light",
            attributes={},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Test Light" in formatted
        assert "OFF" in formatted

    def test_format_switch_state(self, state_input):
        """Test formatting switch state."""
        state = DeviceState(
            entity_id="switch.coffee_maker",
            state="on",
            friendly_name="Coffee Maker",
            attributes={},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Coffee Maker" in formatted
        assert "ON" in formatted

    def test_format_thermostat_state(self, state_input):
        """Test formatting thermostat state."""
        state = DeviceState(
            entity_id="climate.bedroom",
            state="heat",
            friendly_name="Bedroom Thermostat",
            attributes={
                "current_temperature": 20,
                "temperature": 22,
            },
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Bedroom Thermostat" in formatted
        assert "heat" in formatted
        assert "20" in formatted
        assert "22" in formatted

    def test_format_sensor_state(self, state_input):
        """Test formatting sensor state."""
        state = DeviceState(
            entity_id="sensor.temperature",
            state="21.5",
            friendly_name="Temperature Sensor",
            attributes={"unit_of_measurement": "°C"},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Temperature Sensor" in formatted
        assert "21.5" in formatted
        assert "°C" in formatted

    def test_format_binary_sensor_state(self, state_input):
        """Test formatting binary sensor state."""
        state = DeviceState(
            entity_id="binary_sensor.motion",
            state="on",
            friendly_name="Motion Sensor",
            attributes={},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Motion Sensor" in formatted
        assert "detected" in formatted

    def test_format_cover_state(self, state_input):
        """Test formatting cover state."""
        state = DeviceState(
            entity_id="cover.garage",
            state="open",
            friendly_name="Garage Door",
            attributes={"current_position": 100},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Garage Door" in formatted
        assert "open" in formatted
        assert "100%" in formatted

    def test_format_fan_state(self, state_input):
        """Test formatting fan state."""
        state = DeviceState(
            entity_id="fan.bedroom",
            state="on",
            friendly_name="Bedroom Fan",
            attributes={"percentage": 50},
            last_changed="",
        )

        formatted = state_input._format_state_for_llm(state)
        assert "Bedroom Fan" in formatted
        assert "ON" in formatted
        assert "50%" in formatted

    def test_format_all_states(self, state_input):
        """Test formatting all device states."""
        states = [
            DeviceState(
                entity_id="light.test",
                state="on",
                friendly_name="Test Light",
                attributes={"brightness": 255},
                last_changed="",
            ),
            DeviceState(
                entity_id="switch.test",
                state="off",
                friendly_name="Test Switch",
                attributes={},
                last_changed="",
            ),
        ]

        formatted = state_input._format_all_states(states)
        assert "Current smart home device status" in formatted
        assert "Test Light" in formatted
        assert "Test Switch" in formatted

    def test_format_all_states_empty(self, state_input):
        """Test formatting empty state list."""
        formatted = state_input._format_all_states([])
        assert "No smart home devices" in formatted

    def test_get_changed_states(self, state_input):
        """Test detecting changed states."""
        # Set initial state
        state_input._previous_states = {
            "light.test": "off",
            "switch.test": "on",
        }

        current_states = [
            DeviceState(
                entity_id="light.test",
                state="on",  # Changed
                friendly_name="Test Light",
                attributes={},
                last_changed="",
            ),
            DeviceState(
                entity_id="switch.test",
                state="on",  # No change
                friendly_name="Test Switch",
                attributes={},
                last_changed="",
            ),
        ]

        changed = state_input._get_changed_states(current_states)

        assert len(changed) == 1
        assert changed[0].entity_id == "light.test"

    def test_get_changed_states_all_new(self, state_input):
        """Test detecting changes when no previous states."""
        current_states = [
            DeviceState(
                entity_id="light.new",
                state="on",
                friendly_name="New Light",
                attributes={},
                last_changed="",
            ),
        ]

        changed = state_input._get_changed_states(current_states)

        assert len(changed) == 1

    @pytest.mark.asyncio
    async def test_raw_to_text(self, state_input):
        """Test converting raw input to Message."""
        raw = "Test message"
        message = await state_input._raw_to_text(raw)

        assert message is not None
        assert message.message == "Test message"
        assert message.timestamp > 0

    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, state_input):
        """Test handling None input."""
        message = await state_input._raw_to_text(None)
        assert message is None

    def test_formatted_latest_buffer(self, state_input):
        """Test getting formatted latest buffer matches OM1 convention."""
        state_input.messages = [
            Message(timestamp=time.time(), message="Test state info"),
        ]

        # Mock IOProvider to avoid singleton issues in tests
        with patch.object(state_input, "io_provider") as mock_io:
            formatted = state_input.formatted_latest_buffer()

            assert formatted is not None
            assert "\nINPUT: Test Smart Home\n// START\n" in formatted
            assert "Test state info" in formatted
            assert "\n// END\n" in formatted
            mock_io.add_input.assert_called_once()
            # Buffer should be cleared after formatting
            assert state_input.messages == []

    def test_formatted_latest_buffer_empty(self, state_input):
        """Test getting formatted buffer when empty."""
        state_input.messages = []
        formatted = state_input.formatted_latest_buffer()
        assert formatted is None

    @pytest.mark.asyncio
    async def test_poll_respects_interval(self, state_input):
        """Test that polling respects the configured interval."""
        state_input._last_poll_time = time.time()

        with patch.object(
            state_input, "_get_all_states", new_callable=AsyncMock
        ) as mock_get:
            result = await state_input._poll()

            # Should return None without calling _get_all_states
            # because interval hasn't passed
            assert result is None
            mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_executes_after_interval(self, state_input):
        """Test that polling executes after the interval has passed."""
        # Set last poll time far enough in the past
        state_input._last_poll_time = time.time() - state_input.poll_interval - 1

        mock_states = [
            DeviceState(
                entity_id="light.test",
                state="on",
                friendly_name="Test Light",
                attributes={"brightness": 255},
                last_changed="",
            ),
        ]

        with patch.object(
            state_input, "_get_all_states", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_states
            result = await state_input._poll()

            mock_get.assert_called_once()
            assert result is not None

    @pytest.mark.asyncio
    async def test_poll_handles_network_error(self, state_input):
        """Test that polling handles network errors gracefully."""
        state_input._last_poll_time = 0

        with patch.object(
            state_input, "_get_all_states", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = Exception("Network unreachable")
            result = await state_input._poll()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_state_http_error(self, state_input):
        """Test handling of HTTP errors when getting entity state."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response

        with patch.object(
            state_input, "_get_session", new_callable=AsyncMock
        ) as mock_get_session:
            mock_get_session.return_value = mock_session
            result = await state_input._get_entity_state("light.test")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_state_connection_error(self, state_input):
        """Test handling of connection errors when getting entity state."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection refused")

        with patch.object(
            state_input, "_get_session", new_callable=AsyncMock
        ) as mock_get_session:
            mock_get_session.return_value = mock_session
            result = await state_input._get_entity_state("light.test")

            assert result is None
