from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.home_assistant_state import (
    HomeAssistantStateConfig,
    HomeAssistantStateInput,
)


@pytest.fixture(autouse=True)
def reset_provider_singleton():
    """Reset HomeAssistantProvider singleton between tests."""
    from providers.home_assistant_provider import HomeAssistantProvider

    HomeAssistantProvider.reset()  # type: ignore
    yield
    HomeAssistantProvider.reset()  # type: ignore


@pytest.fixture
def config():
    """Create a test config."""
    return HomeAssistantStateConfig(
        base_url="http://ha.local:8123",
        token="test-token",
        token_env="HA_TEST_TOKEN_UNUSED",
        entities=["light.living_room", "sensor.temperature", "climate.thermostat"],
        poll_interval=0.01,
    )


@pytest.fixture
def state_input(config):
    """Create a HomeAssistantStateInput with test config."""
    return HomeAssistantStateInput(config)


def test_init_descriptor(state_input):
    """Test that descriptor_for_LLM is set correctly."""
    assert state_input.descriptor_for_LLM == "Home Status"


def test_init_empty_messages(state_input):
    """Test that messages buffer starts empty."""
    assert state_input.messages == []


def test_init_empty_last_states(state_input):
    """Test that last_states starts empty."""
    assert state_input.last_states == {}


def test_init_empty_entities_warns():
    """Test that empty entity list logs a warning."""
    with patch("inputs.plugins.home_assistant_state.logger") as mock_logger:
        config = HomeAssistantStateConfig(
            token="test-token",
            entities=[],
        )
        HomeAssistantStateInput(config)
    mock_logger.warning.assert_called_once()
    assert "No entities configured" in mock_logger.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_poll_returns_entity_states(state_input):
    """Test that _poll returns entity states from the provider."""
    mock_states = [
        {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light", "brightness": 200},
        },
        {
            "entity_id": "sensor.temperature",
            "state": "22.5",
            "attributes": {
                "friendly_name": "Temperature",
                "unit_of_measurement": "°C",
            },
        },
    ]
    state_input.provider.get_states = AsyncMock(return_value=mock_states)

    result = await state_input._poll()

    assert result is not None
    assert "light.living_room" in result
    assert "sensor.temperature" in result
    assert result["light.living_room"]["state"] == "on"


@pytest.mark.asyncio
async def test_poll_returns_none_on_error(state_input):
    """Test that _poll returns None when provider raises an error."""
    state_input.provider.get_states = AsyncMock(
        side_effect=RuntimeError("Connection failed")
    )

    result = await state_input._poll()
    assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_when_no_entities():
    """Test that _poll returns None when no entities are configured."""
    from providers.home_assistant_provider import HomeAssistantProvider

    HomeAssistantProvider.reset()  # type: ignore

    config = HomeAssistantStateConfig(
        token="test-token",
        entities=[],
        poll_interval=0.01,
    )
    si = HomeAssistantStateInput(config)
    result = await si._poll()
    assert result is None


@pytest.mark.asyncio
async def test_deduplication_same_state_not_reported(state_input):
    """Test that the same state is not reported twice."""
    states = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }

    # First call should report the state
    msg1 = await state_input._raw_to_text(states)
    assert msg1 is not None
    assert "Living Room Light" in msg1.message

    # Second call with same state should return None
    msg2 = await state_input._raw_to_text(states)
    assert msg2 is None


@pytest.mark.asyncio
async def test_state_change_is_reported(state_input):
    """Test that a state change is reported."""
    states_on = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }
    states_off = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "off",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }

    msg1 = await state_input._raw_to_text(states_on)
    assert msg1 is not None

    msg2 = await state_input._raw_to_text(states_off)
    assert msg2 is not None
    assert "off" in msg2.message


def test_format_entity_state_basic(state_input):
    """Test basic entity state formatting."""
    state_data = {
        "entity_id": "sensor.temperature",
        "state": "22.5",
        "attributes": {
            "friendly_name": "Temperature",
            "unit_of_measurement": "°C",
        },
    }
    result = state_input._format_entity_state(state_data)
    assert "Temperature: 22.5 °C" in result


def test_format_entity_state_light_with_brightness(state_input):
    """Test light entity formatting includes brightness percentage."""
    state_data = {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {
            "friendly_name": "Living Room Light",
            "brightness": 255,
        },
    }
    result = state_input._format_entity_state(state_data)
    assert "Living Room Light: on" in result
    assert "brightness 100%" in result


def test_format_entity_state_light_off_no_brightness(state_input):
    """Test light entity formatting does not include brightness when off."""
    state_data = {
        "entity_id": "light.living_room",
        "state": "off",
        "attributes": {
            "friendly_name": "Living Room Light",
            "brightness": 0,
        },
    }
    result = state_input._format_entity_state(state_data)
    assert "Living Room Light: off" in result
    assert "brightness" not in result


def test_format_entity_state_climate_with_temperature(state_input):
    """Test climate entity formatting includes current temperature."""
    state_data = {
        "entity_id": "climate.thermostat",
        "state": "heat",
        "attributes": {
            "friendly_name": "Thermostat",
            "current_temperature": 21.0,
        },
    }
    result = state_input._format_entity_state(state_data)
    assert "Thermostat: heat" in result
    assert "current temperature 21.0" in result


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_messages(state_input):
    """Test that raw_to_text appends new messages to the buffer."""
    states = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }

    await state_input.raw_to_text(states)
    assert len(state_input.messages) == 1


@pytest.mark.asyncio
async def test_raw_to_text_none_input_no_append(state_input):
    """Test that None input does not append to messages."""
    await state_input.raw_to_text(None)
    assert len(state_input.messages) == 0


def test_formatted_latest_buffer_empty_returns_none(state_input):
    """Test that empty buffer returns None."""
    assert state_input.formatted_latest_buffer() is None


@pytest.mark.asyncio
async def test_formatted_latest_buffer_format(state_input):
    """Test the output format of formatted_latest_buffer."""
    states = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }

    await state_input.raw_to_text(states)
    result = state_input.formatted_latest_buffer()

    assert result is not None
    assert "INPUT: Home Status" in result
    assert "// START" in result
    assert "// END" in result
    assert "Living Room Light: on" in result


@pytest.mark.asyncio
async def test_formatted_latest_buffer_clears_messages(state_input):
    """Test that formatted_latest_buffer clears the message buffer."""
    states = {
        "light.living_room": {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room Light"},
        },
    }

    await state_input.raw_to_text(states)
    state_input.formatted_latest_buffer()

    assert len(state_input.messages) == 0
    assert state_input.formatted_latest_buffer() is None
