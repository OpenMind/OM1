"""Tests for Home Assistant WebSocket action connector."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from actions.home_assistant.connector.websocket import (
    COLOR_MAP,
    HomeAssistantWebSocketConfig,
    HomeAssistantWebSocketConnector,
)
from actions.home_assistant.interface import (
    HAAction,
    HADeviceType,
    HomeAssistantInput,
)


def make_connector(base_url="http://ha.local:8123", token="test_token"):
    """Helper to create a connector instance."""
    config = HomeAssistantWebSocketConfig(base_url=base_url, token=token, timeout=5.0)
    return HomeAssistantWebSocketConnector(config)


def make_ws_mock(auth_ok=True, command_success=True, unexpected_type=False):
    """Helper to create a mock WebSocket connection."""
    auth_required = json.dumps({"type": "auth_required"})
    if unexpected_type:
        auth_required = json.dumps({"type": "hello"})

    if auth_ok:
        auth_response = json.dumps({"type": "auth_ok"})
    else:
        auth_response = json.dumps({"type": "auth_invalid"})

    if command_success:
        command_response = json.dumps({"id": 1, "type": "result", "success": True})
    else:
        command_response = json.dumps(
            {
                "id": 1,
                "type": "result",
                "success": False,
                "error": {"code": "unknown_error", "message": "failed"},
            }
        )

    ws = AsyncMock()
    ws.recv = AsyncMock(side_effect=[auth_required, auth_response, command_response])
    ws.send = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)
    return ws


class TestHomeAssistantWebSocketConfig:
    """Tests for HomeAssistantWebSocketConfig."""

    def test_default_values(self):
        """Test config default values."""
        config = HomeAssistantWebSocketConfig()
        assert config.base_url == ""
        assert config.token == ""
        assert config.timeout == 10.0

    def test_custom_values(self):
        """Test config with custom values."""
        config = HomeAssistantWebSocketConfig(
            base_url="http://ha.local:8123",
            token="my_token",
            timeout=30.0,
        )
        assert config.base_url == "http://ha.local:8123"
        assert config.token == "my_token"
        assert config.timeout == 30.0


class TestHomeAssistantWebSocketConnectorInit:
    """Tests for connector initialization."""

    def test_http_url_converted_to_ws(self):
        """Test that http:// is converted to ws://."""
        connector = make_connector(base_url="http://ha.local:8123")
        assert connector._ws_url == "ws://ha.local:8123/api/websocket"

    def test_https_url_converted_to_wss(self):
        """Test that https:// is converted to wss://."""
        connector = make_connector(base_url="https://ha.local:8123")
        assert connector._ws_url == "wss://ha.local:8123/api/websocket"

    def test_trailing_slash_stripped(self):
        """Test that trailing slash is stripped."""
        connector = make_connector(base_url="http://ha.local:8123/")
        assert connector._ws_url == "ws://ha.local:8123/api/websocket"

    def test_no_scheme_url(self):
        """Test URL without scheme."""
        connector = make_connector(base_url="ha.local:8123")
        assert connector._ws_url == "ha.local:8123/api/websocket"

    def test_warns_missing_base_url(self):
        """Test warning when base_url is missing."""
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            make_connector(base_url="")
            assert any("base_url" in str(c) for c in mock_warn.call_args_list)

    def test_warns_missing_token(self):
        """Test warning when token is missing."""
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            make_connector(token="")
            assert any("token" in str(c) for c in mock_warn.call_args_list)


class TestSendCommand:
    """Tests for _send_command()."""

    @pytest.mark.asyncio
    async def test_send_command_success(self):
        """Test successful command send."""
        connector = make_connector()
        ws = make_ws_mock(auth_ok=True, command_success=True)
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect",
            return_value=ws,
        ):
            result = await connector._send_command("light", "turn_on", "light.x")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_command_no_token(self):
        """Test that missing token returns False."""
        connector = make_connector(token="")
        result = await connector._send_command("light", "turn_on", "light.x")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_command_auth_failed(self):
        """Test handling of auth failure."""
        connector = make_connector()
        ws = make_ws_mock(auth_ok=False)
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect",
            return_value=ws,
        ):
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any(
                    "authentication failed" in str(c) for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_send_command_unexpected_auth_type(self):
        """Test handling of unexpected initial message type."""
        connector = make_connector()
        ws = make_ws_mock(unexpected_type=True)
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect",
            return_value=ws,
        ):
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any("auth_required" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_send_command_failed_response(self):
        """Test handling of failed command response."""
        connector = make_connector()
        ws = make_ws_mock(command_success=False)
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect",
            return_value=ws,
        ):
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any("command failed" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_send_command_timeout(self):
        """Test handling of timeout error."""
        connector = make_connector()
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect"
        ) as mock_connect:
            mock_connect.side_effect = TimeoutError()
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any("timed out" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_send_command_websocket_exception(self):
        """Test handling of WebSocket exception."""
        import websockets.exceptions

        connector = make_connector()
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect"
        ) as mock_connect:
            mock_connect.side_effect = websockets.exceptions.WebSocketException("err")
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any("WebSocket error" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_send_command_unexpected_exception(self):
        """Test handling of unexpected exception."""
        connector = make_connector()
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect"
        ) as mock_connect:
            mock_connect.side_effect = RuntimeError("boom")
            with patch(
                "actions.home_assistant.connector.websocket.logging.error"
            ) as mock_err:
                result = await connector._send_command("light", "turn_on", "light.x")
                assert result is False
                assert any(
                    "unexpected error" in str(c) for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_send_command_increments_msg_id(self):
        """Test that message ID increments on each call."""
        connector = make_connector()
        assert connector._msg_id == 1
        ws = make_ws_mock()
        with patch(
            "actions.home_assistant.connector.websocket.websockets.connect",
            return_value=ws,
        ):
            await connector._send_command("light", "turn_on", "light.x")
        assert connector._msg_id == 2


class TestConnectLight:
    """Tests for light device control."""

    @pytest.mark.asyncio
    async def test_light_turn_on(self):
        """Test light turn on."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT, entity_id="light.x", action=HAAction.TURN_ON
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("light", "turn_on", "light.x")

    @pytest.mark.asyncio
    async def test_light_turn_off(self):
        """Test light turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.TURN_OFF,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("light", "turn_off", "light.x")

    @pytest.mark.asyncio
    async def test_light_set_brightness(self):
        """Test light set brightness."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.SET_BRIGHTNESS,
            brightness=128,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with(
                "light", "turn_on", "light.x", {"brightness": 128}
            )

    @pytest.mark.asyncio
    async def test_light_set_color(self):
        """Test light set color."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.SET_COLOR,
            color="red",
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with(
                "light", "turn_on", "light.x", {"hs_color": COLOR_MAP["red"]}
            )

    @pytest.mark.asyncio
    async def test_light_unknown_color_defaults_to_white(self):
        """Test that unknown color defaults to white."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.SET_COLOR,
            color="magenta",
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with(
                "light", "turn_on", "light.x", {"hs_color": COLOR_MAP["white"]}
            )

    @pytest.mark.asyncio
    async def test_light_unsupported_action_warns(self):
        """Test warning for unsupported light action."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.x",
            action=HAAction.SET_TEMPERATURE,
        )
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            await connector.connect(inp)
            assert any("not supported" in str(c) for c in mock_warn.call_args_list)


class TestConnectSwitch:
    """Tests for switch device control."""

    @pytest.mark.asyncio
    async def test_switch_turn_on(self):
        """Test switch turn on."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_ON,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("switch", "turn_on", "switch.fan")

    @pytest.mark.asyncio
    async def test_switch_turn_off(self):
        """Test switch turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_OFF,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("switch", "turn_off", "switch.fan")

    @pytest.mark.asyncio
    async def test_switch_unsupported_action_warns(self):
        """Test warning for unsupported switch action."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.SET_TEMPERATURE,
        )
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            await connector.connect(inp)
            assert any("not supported" in str(c) for c in mock_warn.call_args_list)


class TestConnectClimate:
    """Tests for climate device control."""

    @pytest.mark.asyncio
    async def test_climate_set_temperature(self):
        """Test climate set temperature."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.SET_TEMPERATURE,
            temperature=24.0,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with(
                "climate", "set_temperature", "climate.bedroom", {"temperature": 24.0}
            )

    @pytest.mark.asyncio
    async def test_climate_turn_on(self):
        """Test climate turn on."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.TURN_ON,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("climate", "turn_on", "climate.bedroom")

    @pytest.mark.asyncio
    async def test_climate_turn_off(self):
        """Test climate turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.TURN_OFF,
        )
        with patch.object(
            connector, "_send_command", new_callable=AsyncMock
        ) as mock_cmd:
            await connector.connect(inp)
            mock_cmd.assert_called_once_with("climate", "turn_off", "climate.bedroom")

    @pytest.mark.asyncio
    async def test_climate_unsupported_action_warns(self):
        """Test warning for unsupported climate action."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.SET_BRIGHTNESS,
        )
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            await connector.connect(inp)
            assert any("not supported" in str(c) for c in mock_warn.call_args_list)


class TestConnectUnsupportedDeviceType:
    """Tests for unsupported device type."""

    @pytest.mark.asyncio
    async def test_unsupported_device_type_warns(self):
        """Test warning for unsupported device type."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT, entity_id="light.x", action=HAAction.TURN_ON
        )
        with patch(
            "actions.home_assistant.connector.websocket.logging.warning"
        ) as mock_warn:
            with patch(
                "actions.home_assistant.connector.websocket.HADeviceType"
            ) as mock_enum:
                mock_enum.LIGHT = "FAKE"
                mock_enum.SWITCH = "FAKE"
                mock_enum.CLIMATE = "FAKE"
                await connector.connect(inp)
                assert any("not supported" in str(c) for c in mock_warn.call_args_list)
