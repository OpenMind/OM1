import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from actions.home_assistant.connector.rest_api import (
    COLOR_MAP,
    HomeAssistantConfig,
    HomeAssistantRESTConnector,
)
from actions.home_assistant.interface import (
    HAAction,
    HADeviceType,
    HomeAssistant,
    HomeAssistantInput,
)


class TestHomeAssistantInput:
    def test_default_values(self):
        inp = HomeAssistantInput()
        assert inp.device_type == HADeviceType.LIGHT
        assert inp.entity_id == ""
        assert inp.action == HAAction.TURN_ON
        assert inp.brightness == 255
        assert inp.color == ""
        assert inp.temperature == 22.0

    def test_custom_values(self):
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_OFF,
        )
        assert inp.device_type == HADeviceType.SWITCH
        assert inp.entity_id == "switch.fan"
        assert inp.action == HAAction.TURN_OFF

    def test_interface_creation(self):
        inp = HomeAssistantInput(entity_id="light.living_room")
        iface = HomeAssistant(input=inp, output=inp)
        assert iface.input.entity_id == "light.living_room"


class TestHomeAssistantConfig:
    def test_defaults(self):
        config = HomeAssistantConfig()
        assert config.base_url == ""
        assert config.token == ""
        assert config.timeout == 10.0

    def test_custom_values(self):
        config = HomeAssistantConfig(
            base_url="http://ha.local:8123",
            token="abc123",
            timeout=5.0,
        )
        assert config.base_url == "http://ha.local:8123"
        assert config.token == "abc123"
        assert config.timeout == 5.0


class TestHomeAssistantRESTConnectorInit:
    def test_init_warns_missing_base_url(self):
        with patch(
            "actions.home_assistant.connector.rest_api.logging.warning"
        ) as mock_warn:
            HomeAssistantRESTConnector(HomeAssistantConfig(token="tok"))
            warned_msgs = [str(c) for c in mock_warn.call_args_list]
            assert any("base_url" in m for m in warned_msgs)

    def test_init_warns_missing_token(self):
        with patch(
            "actions.home_assistant.connector.rest_api.logging.warning"
        ) as mock_warn:
            HomeAssistantRESTConnector(
                HomeAssistantConfig(base_url="http://ha.local:8123")
            )
            warned_msgs = [str(c) for c in mock_warn.call_args_list]
            assert any("token" in m for m in warned_msgs)

    def test_init_no_warning_when_configured(self):
        with patch(
            "actions.home_assistant.connector.rest_api.logging.warning"
        ) as mock_warn:
            HomeAssistantRESTConnector(
                HomeAssistantConfig(base_url="http://ha.local:8123", token="tok")
            )
            mock_warn.assert_not_called()


def make_connector(base_url="http://ha.local:8123", token="test-token"):
    return HomeAssistantRESTConnector(
        HomeAssistantConfig(base_url=base_url, token=token)
    )


def mock_ha_session(status=200, json_data=None):
    """Return a patch context for aiohttp.ClientSession."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data or [])
    mock_response.text = AsyncMock(return_value="error text")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock()
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return (
        patch(
            "actions.home_assistant.connector.rest_api.aiohttp.ClientSession",
            return_value=mock_session,
        ),
        mock_session,
    )


class TestConnectMissingConfig:
    @pytest.mark.asyncio
    async def test_skips_empty_entity_id(self):
        connector = make_connector()
        inp = HomeAssistantInput(entity_id="")
        with patch(
            "actions.home_assistant.connector.rest_api.logging.warning"
        ) as mock_warn:
            await connector.connect(inp)
            mock_warn.assert_called_once()
            assert "entity_id" in str(mock_warn.call_args)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_base_url(self):
        connector = HomeAssistantRESTConnector(
            HomeAssistantConfig(base_url="", token="tok")
        )
        inp = HomeAssistantInput(entity_id="light.x")
        with patch(
            "actions.home_assistant.connector.rest_api.logging.error"
        ) as mock_err:
            await connector.connect(inp)
            assert any(
                "base_url" in str(c) or "token" in str(c)
                for c in mock_err.call_args_list
            )


class TestConnectLight:
    @pytest.mark.asyncio
    async def test_light_turn_on(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.TURN_ON,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            mock_session.post.assert_called_once()
            url = mock_session.post.call_args[0][0]
            assert "light/turn_on" in url
            payload = mock_session.post.call_args[1]["json"]
            assert payload["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_light_turn_off(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.TURN_OFF,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "light/turn_off" in url

    @pytest.mark.asyncio
    async def test_light_set_brightness(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.bedroom",
            action=HAAction.SET_BRIGHTNESS,
            brightness=128,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "light/turn_on" in url
            payload = mock_session.post.call_args[1]["json"]
            assert payload["brightness"] == 128

    @pytest.mark.asyncio
    async def test_light_brightness_clamped(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.bedroom",
            action=HAAction.SET_BRIGHTNESS,
            brightness=999,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            payload = mock_session.post.call_args[1]["json"]
            assert payload["brightness"] == 255

    @pytest.mark.asyncio
    async def test_light_set_color_red(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.kitchen",
            action=HAAction.SET_COLOR,
            color="red",
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            payload = mock_session.post.call_args[1]["json"]
            assert payload["hs_color"] == COLOR_MAP["red"]

    @pytest.mark.asyncio
    async def test_light_set_unknown_color_defaults_white(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.kitchen",
            action=HAAction.SET_COLOR,
            color="chartreuse",
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            payload = mock_session.post.call_args[1]["json"]
            assert payload["hs_color"] == COLOR_MAP["white"]


class TestConnectSwitch:
    @pytest.mark.asyncio
    async def test_switch_turn_on(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_ON,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "switch/turn_on" in url

    @pytest.mark.asyncio
    async def test_switch_turn_off(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_OFF,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "switch/turn_off" in url

    @pytest.mark.asyncio
    async def test_switch_unsupported_action_warns(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.SET_BRIGHTNESS,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            with patch(
                "actions.home_assistant.connector.rest_api.logging.warning"
            ) as mock_warn:
                await connector.connect(inp)
                assert any("not supported" in str(c) for c in mock_warn.call_args_list)


class TestConnectClimate:
    @pytest.mark.asyncio
    async def test_climate_set_temperature(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.SET_TEMPERATURE,
            temperature=24.5,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "climate/set_temperature" in url
            payload = mock_session.post.call_args[1]["json"]
            assert payload["temperature"] == 24.5

    @pytest.mark.asyncio
    async def test_climate_turn_on(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.TURN_ON,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "climate/turn_on" in url


class TestNetworkErrors:
    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_ON,
        )
        with patch(
            "actions.home_assistant.connector.rest_api.aiohttp.ClientSession"
        ) as mock_cls:
            mock_cls.side_effect = asyncio.TimeoutError()
            with patch(
                "actions.home_assistant.connector.rest_api.logging.error"
            ) as mock_err:
                await connector.connect(inp)
                assert any(
                    "timed out" in str(c) or "timeout" in str(c).lower()
                    for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_handles_client_error(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_ON,
        )
        with patch(
            "actions.home_assistant.connector.rest_api.aiohttp.ClientSession"
        ) as mock_cls:
            mock_cls.side_effect = aiohttp.ClientError("connection refused")
            with patch(
                "actions.home_assistant.connector.rest_api.logging.error"
            ) as mock_err:
                await connector.connect(inp)
                assert any(
                    "network error" in str(c).lower() for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_handles_error_status(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_ON,
        )
        ctx, _ = mock_ha_session(status=401)
        with ctx:
            with patch(
                "actions.home_assistant.connector.rest_api.logging.error"
            ) as mock_err:
                await connector.connect(inp)
                assert any("401" in str(c) for c in mock_err.call_args_list)


class TestCoverageGaps:
    @pytest.mark.asyncio
    async def test_call_service_unexpected_exception(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_ON,
        )
        with patch(
            "actions.home_assistant.connector.rest_api.aiohttp.ClientSession"
        ) as mock_cls:
            mock_cls.side_effect = RuntimeError("unexpected boom")
            with patch(
                "actions.home_assistant.connector.rest_api.logging.error"
            ) as mock_err:
                await connector.connect(inp)
                assert any(
                    "unexpected error" in str(c).lower()
                    for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_light_unsupported_action_warns(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.SET_TEMPERATURE,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            with patch(
                "actions.home_assistant.connector.rest_api.logging.warning"
            ) as mock_warn:
                await connector.connect(inp)
                assert any("not supported" in str(c) for c in mock_warn.call_args_list)

    @pytest.mark.asyncio
    async def test_climate_unsupported_action_warns(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.SET_BRIGHTNESS,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            with patch(
                "actions.home_assistant.connector.rest_api.logging.warning"
            ) as mock_warn:
                await connector.connect(inp)
                assert any("not supported" in str(c) for c in mock_warn.call_args_list)

    @pytest.mark.asyncio
    async def test_climate_turn_off(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.TURN_OFF,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            await connector.connect(inp)
            url = mock_session.post.call_args[0][0]
            assert "climate/turn_off" in url

    @pytest.mark.asyncio
    async def test_unsupported_device_type_warns(self):
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_ON,
        )
        ctx, mock_session = mock_ha_session()
        with ctx:
            with patch(
                "actions.home_assistant.connector.rest_api.logging.warning"
            ) as mock_warn:
                # Patch _call_service agar tidak dipanggil, lalu paksa masuk else branch
                with patch.object(connector, "_call_service") as _:
                    # Simulasi device_type yang tidak cocok dengan semua kondisi
                    # dengan mocking langsung di level connect()
                    with patch(
                        "actions.home_assistant.connector.rest_api.HADeviceType"
                    ) as mock_enum:
                        mock_enum.LIGHT = "FAKE_LIGHT"
                        mock_enum.SWITCH = "FAKE_SWITCH"
                        mock_enum.CLIMATE = "FAKE_CLIMATE"
                        await connector.connect(inp)
                        assert any(
                            "not supported" in str(c) for c in mock_warn.call_args_list
                        )
