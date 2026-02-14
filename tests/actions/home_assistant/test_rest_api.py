from unittest.mock import AsyncMock, patch

import pytest

from actions.home_assistant.connector.rest_api import (
    HomeAssistantConfig,
    HomeAssistantRESTConnector,
    _parse_rgb_color,
)
from actions.home_assistant.interface import HomeAssistantInput


@pytest.fixture(autouse=True)
def reset_provider_singleton():
    """Reset HomeAssistantProvider singleton between tests."""
    from providers.home_assistant_provider import HomeAssistantProvider

    HomeAssistantProvider.reset()  # type: ignore
    yield
    HomeAssistantProvider.reset()  # type: ignore


@pytest.fixture
def connector():
    """Create a HomeAssistantRESTConnector with all device types."""
    config = HomeAssistantConfig(
        base_url="http://ha.local:8123",
        token="test-token",
        token_env="HA_TEST_TOKEN_UNUSED",
        devices={
            "living_room_light": "light.living_room",
            "thermostat": "climate.thermostat",
            "garage_fan": "fan.garage",
            "kitchen_switch": "switch.kitchen",
            "front_door": "lock.front_door",
            "blinds": "cover.blinds",
            "tv": "media_player.tv",
            "roomba": "vacuum.roomba",
            "movie_mode": "scene.movie_mode",
            "alarm": "alarm_control_panel.home",
            "morning_routine": "script.morning_routine",
        },
    )
    return HomeAssistantRESTConnector(config)


# --- Universal commands (on/off/toggle) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("on", "turn_on"),
        ("off", "turn_off"),
        ("toggle", "toggle"),
    ],
)
async def test_basic_commands(connector, command, expected_service):
    """Test on, off, toggle commands call the correct HA service."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="living_room_light", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service=expected_service,
        entity_id="light.living_room",
    )


@pytest.mark.asyncio
async def test_command_case_insensitive(connector):
    """Test that commands are case-insensitive."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="living_room_light", command="  ON  ")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service="turn_on",
        entity_id="light.living_room",
    )


# --- Legacy "set" command ---


@pytest.mark.asyncio
async def test_set_light_brightness(connector):
    """Test that legacy set on a light sends brightness_pct."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set", value=75.0
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service="turn_on",
        entity_id="light.living_room",
        brightness_pct=75.0,
    )


@pytest.mark.asyncio
async def test_set_climate_temperature(connector):
    """Test that legacy set on a climate entity sends temperature."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="thermostat", command="set", value=22.5)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="climate",
        service="set_temperature",
        entity_id="climate.thermostat",
        temperature=22.5,
    )


@pytest.mark.asyncio
async def test_set_generic_domain(connector):
    """Test that legacy set on a non-light/non-climate domain sends set_value."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="garage_fan", command="set", value=50.0)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="fan",
        service="set_value",
        entity_id="fan.garage",
        value=50.0,
    )


# --- Light commands ---


@pytest.mark.asyncio
async def test_set_brightness_command(connector):
    """Test set_brightness sends brightness_pct."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set_brightness", value=80.0
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service="turn_on",
        entity_id="light.living_room",
        brightness_pct=80.0,
    )


@pytest.mark.asyncio
async def test_set_color_command(connector):
    """Test set_color sends rgb_color."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set_color", mode="#FF5500"
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service="turn_on",
        entity_id="light.living_room",
        rgb_color=[255, 85, 0],
    )


@pytest.mark.asyncio
async def test_set_color_temp_command(connector):
    """Test set_color_temp sends color_temp_kelvin."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set_color_temp", value=4000.0
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="light",
        service="turn_on",
        entity_id="light.living_room",
        color_temp_kelvin=4000.0,
    )


@pytest.mark.asyncio
async def test_set_color_without_mode_logs_error(connector):
    """Test set_color without mode logs error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set_color", mode=None
    )

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "requires a mode" in mock_logger.error.call_args[0][0]
    connector.provider.call_service.assert_not_called()


# --- Climate commands ---


@pytest.mark.asyncio
async def test_set_temperature_command(connector):
    """Test set_temperature sends temperature."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="thermostat", command="set_temperature", value=24.0
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="climate",
        service="set_temperature",
        entity_id="climate.thermostat",
        temperature=24.0,
    )


@pytest.mark.asyncio
async def test_set_hvac_mode_command(connector):
    """Test set_hvac_mode sends hvac_mode."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="thermostat", command="set_hvac_mode", mode="cool"
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="climate",
        service="set_hvac_mode",
        entity_id="climate.thermostat",
        hvac_mode="cool",
    )


@pytest.mark.asyncio
async def test_set_fan_mode_command(connector):
    """Test set_fan_mode sends fan_mode."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="thermostat", command="set_fan_mode", mode="high"
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="climate",
        service="set_fan_mode",
        entity_id="climate.thermostat",
        fan_mode="high",
    )


@pytest.mark.asyncio
async def test_set_preset_command(connector):
    """Test set_preset sends preset_mode."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="thermostat", command="set_preset", mode="eco"
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="climate",
        service="set_preset_mode",
        entity_id="climate.thermostat",
        preset_mode="eco",
    )


# --- Lock commands ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("lock", "lock"),
        ("unlock", "unlock"),
    ],
)
async def test_lock_commands(connector, command, expected_service):
    """Test lock and unlock commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="front_door", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="lock",
        service=expected_service,
        entity_id="lock.front_door",
    )


# --- Cover commands ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("open", "open_cover"),
        ("close", "close_cover"),
        ("stop", "stop_cover"),
    ],
)
async def test_cover_commands(connector, command, expected_service):
    """Test cover open, close, stop commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="blinds", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="cover",
        service=expected_service,
        entity_id="cover.blinds",
    )


@pytest.mark.asyncio
async def test_cover_set_position(connector):
    """Test cover set_position sends position."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="blinds", command="set_position", value=50.0)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="cover",
        service="set_cover_position",
        entity_id="cover.blinds",
        position=50.0,
    )


# --- Media player commands ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("play", "media_play"),
        ("pause", "media_pause"),
        ("media_stop", "media_stop"),
    ],
)
async def test_media_basic_commands(connector, command, expected_service):
    """Test media player basic commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="tv", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="media_player",
        service=expected_service,
        entity_id="media_player.tv",
    )


@pytest.mark.asyncio
async def test_media_volume_set(connector):
    """Test volume_set converts percentage to 0-1 range."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="tv", command="volume_set", value=75.0)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="media_player",
        service="volume_set",
        entity_id="media_player.tv",
        volume_level=0.75,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_muted",
    [
        ("volume_mute", True),
        ("volume_unmute", False),
    ],
)
async def test_media_volume_mute(connector, command, expected_muted):
    """Test volume mute/unmute commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="tv", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="media_player",
        service="volume_mute",
        entity_id="media_player.tv",
        is_volume_muted=expected_muted,
    )


@pytest.mark.asyncio
async def test_media_select_source(connector):
    """Test select_source sends source name."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="tv", command="select_source", mode="HDMI 1")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="media_player",
        service="select_source",
        entity_id="media_player.tv",
        source="HDMI 1",
    )


# --- Fan commands ---


@pytest.mark.asyncio
async def test_fan_set_percentage(connector):
    """Test fan set_percentage sends percentage."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="garage_fan", command="set_percentage", value=60.0
    )
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="fan",
        service="set_percentage",
        entity_id="fan.garage",
        percentage=60.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_oscillating",
    [
        ("oscillate", True),
        ("stop_oscillate", False),
    ],
)
async def test_fan_oscillate(connector, command, expected_oscillating):
    """Test fan oscillate on/off commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="garage_fan", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="fan",
        service="oscillate",
        entity_id="fan.garage",
        oscillating=expected_oscillating,
    )


# --- Vacuum commands ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("start", "start"),
        ("stop", "stop"),
        ("vacuum_pause", "pause"),
        ("return_to_base", "return_to_base"),
    ],
)
async def test_vacuum_commands(connector, command, expected_service):
    """Test vacuum commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="roomba", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="vacuum",
        service=expected_service,
        entity_id="vacuum.roomba",
    )


# --- Scene commands ---


@pytest.mark.asyncio
async def test_scene_activate(connector):
    """Test scene activate calls turn_on."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="movie_mode", command="activate")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="scene",
        service="turn_on",
        entity_id="scene.movie_mode",
    )


# --- Alarm commands ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected_service",
    [
        ("arm_home", "alarm_arm_home"),
        ("arm_away", "alarm_arm_away"),
        ("arm_night", "alarm_arm_night"),
        ("disarm", "alarm_disarm"),
    ],
)
async def test_alarm_commands(connector, command, expected_service):
    """Test alarm commands."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="alarm", command=command)
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="alarm_control_panel",
        service=expected_service,
        entity_id="alarm_control_panel.home",
    )


# --- Script commands ---


@pytest.mark.asyncio
async def test_script_run(connector):
    """Test script run calls turn_on."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="morning_routine", command="run")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="script",
        service="turn_on",
        entity_id="script.morning_routine",
    )


# --- Domain-aware "stop" command ---


@pytest.mark.asyncio
async def test_stop_cover_domain(connector):
    """Test stop on cover domain calls stop_cover."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="blinds", command="stop")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="cover",
        service="stop_cover",
        entity_id="cover.blinds",
    )


@pytest.mark.asyncio
async def test_stop_vacuum_domain(connector):
    """Test stop on vacuum domain calls stop."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="roomba", command="stop")
    await connector.connect(input_data)

    connector.provider.call_service.assert_called_once_with(
        domain="vacuum",
        service="stop",
        entity_id="vacuum.roomba",
    )


# --- Error cases ---


@pytest.mark.asyncio
async def test_unknown_device_logs_error(connector):
    """Test that an unknown device alias logs an error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="nonexistent_device", command="on")

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "Unknown device alias" in mock_logger.error.call_args[0][0]
    connector.provider.call_service.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_command_logs_error(connector):
    """Test that an unknown command logs an error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(device="living_room_light", command="dance")

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "Unknown command" in mock_logger.error.call_args[0][0]
    connector.provider.call_service.assert_not_called()


@pytest.mark.asyncio
async def test_set_without_value_logs_error(connector):
    """Test that 'set' without a value logs an error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set", value=None
    )

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "requires a numeric value" in mock_logger.error.call_args[0][0]
    connector.provider.call_service.assert_not_called()


@pytest.mark.asyncio
async def test_set_brightness_without_value_logs_error(connector):
    """Test that set_brightness without value logs error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="living_room_light", command="set_brightness", value=None
    )

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "requires a numeric value" in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_set_hvac_mode_without_mode_logs_error(connector):
    """Test that set_hvac_mode without mode logs error."""
    connector.provider.call_service = AsyncMock()

    input_data = HomeAssistantInput(
        device="thermostat", command="set_hvac_mode", mode=None
    )

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "requires a mode" in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_provider_runtime_error_logged(connector):
    """Test that provider RuntimeError is caught and logged."""
    connector.provider.call_service = AsyncMock(side_effect=RuntimeError("HTTP 500"))

    input_data = HomeAssistantInput(device="living_room_light", command="on")

    with patch("actions.home_assistant.connector.rest_api.logger") as mock_logger:
        await connector.connect(input_data)

    mock_logger.error.assert_called_once()
    assert "HTTP 500" in mock_logger.error.call_args[0][0]


# --- Entity resolution ---


def test_resolve_entity_success(connector):
    """Test successful entity resolution from alias."""
    assert connector._resolve_entity("living_room_light") == "light.living_room"
    assert connector._resolve_entity("thermostat") == "climate.thermostat"
    assert connector._resolve_entity("front_door") == "lock.front_door"
    assert connector._resolve_entity("tv") == "media_player.tv"


def test_resolve_entity_unknown_raises(connector):
    """Test that unknown alias raises ValueError."""
    with pytest.raises(ValueError, match="Unknown device alias"):
        connector._resolve_entity("nonexistent")


# --- RGB color parsing ---


def test_parse_rgb_color_with_hash():
    """Test parsing hex color with # prefix."""
    assert _parse_rgb_color("#FF0000") == [255, 0, 0]
    assert _parse_rgb_color("#00FF00") == [0, 255, 0]
    assert _parse_rgb_color("#0000FF") == [0, 0, 255]


def test_parse_rgb_color_without_hash():
    """Test parsing hex color without # prefix."""
    assert _parse_rgb_color("FF5500") == [255, 85, 0]


def test_parse_rgb_color_invalid_length():
    """Test that invalid hex length raises ValueError."""
    with pytest.raises(ValueError, match="Invalid color hex"):
        _parse_rgb_color("#FFF")


def test_parse_rgb_color_mixed_case():
    """Test parsing hex color with mixed case."""
    assert _parse_rgb_color("#ff8800") == [255, 136, 0]
    assert _parse_rgb_color("#Ff8800") == [255, 136, 0]


# --- Service call resolution (direct unit tests) ---


def test_get_service_call_on():
    """Test on command returns turn_on service."""
    config = HomeAssistantConfig(token="t", devices={})
    c = HomeAssistantRESTConnector(config)
    service, data = c._get_service_call("light", "on", None, None)
    assert service == "turn_on"
    assert data == {}


def test_get_service_call_set_without_value_raises():
    """Test set command without value raises ValueError."""
    config = HomeAssistantConfig(token="t", devices={})
    c = HomeAssistantRESTConnector(config)
    with pytest.raises(ValueError, match="requires a numeric value"):
        c._get_service_call("light", "set", None, None)


def test_get_service_call_unknown_command_raises():
    """Test unknown command raises ValueError."""
    config = HomeAssistantConfig(token="t", devices={})
    c = HomeAssistantRESTConnector(config)
    with pytest.raises(ValueError, match="Unknown command"):
        c._get_service_call("light", "dance", None, None)
