"""Tests for Home Assistant MQTT action connector."""

from unittest.mock import AsyncMock, patch

import pytest

from actions.home_assistant.connector.mqtt import (
    COLOR_MAP,
    HomeAssistantMQTTConfig,
    HomeAssistantMQTTConnector,
)
from actions.home_assistant.interface import (
    HAAction,
    HADeviceType,
    HomeAssistantInput,
)


def make_connector(broker="mqtt.local", port=1883):
    """Helper to create a connector instance."""
    config = HomeAssistantMQTTConfig(
        broker=broker,
        port=port,
        username="user",
        password="pass",
        topic_prefix="homeassistant",
        timeout=5.0,
    )
    return HomeAssistantMQTTConnector(config)


def mock_mqtt_session():
    """Helper to mock aiomqtt.Client context manager."""
    mock_client = AsyncMock()
    mock_client.publish = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    ctx = patch(
        "actions.home_assistant.connector.mqtt.aiomqtt.Client",
        return_value=mock_client,
    )
    return ctx, mock_client


class TestHomeAssistantMQTTConfig:
    """Tests for HomeAssistantMQTTConfig."""

    def test_default_values(self):
        """Test config default values."""
        config = HomeAssistantMQTTConfig()
        assert config.broker == ""
        assert config.port == 1883
        assert config.username == ""
        assert config.password == ""
        assert config.topic_prefix == "homeassistant"
        assert config.timeout == 10.0

    def test_custom_values(self):
        """Test config with custom values."""
        config = HomeAssistantMQTTConfig(
            broker="192.168.1.100",
            port=8883,
            username="admin",
            password="secret",
            topic_prefix="ha",
            timeout=30.0,
        )
        assert config.broker == "192.168.1.100"
        assert config.port == 8883
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.topic_prefix == "ha"
        assert config.timeout == 30.0


class TestHomeAssistantMQTTConnectorInit:
    """Tests for connector initialization."""

    def test_init_stores_config(self):
        """Test that config values are stored correctly."""
        connector = make_connector()
        assert connector._broker == "mqtt.local"
        assert connector._port == 1883
        assert connector._topic_prefix == "homeassistant"

    def test_empty_username_becomes_none(self):
        """Test that empty username is converted to None."""
        config = HomeAssistantMQTTConfig(broker="mqtt.local")
        connector = HomeAssistantMQTTConnector(config)
        assert connector._username is None

    def test_empty_password_becomes_none(self):
        """Test that empty password is converted to None."""
        config = HomeAssistantMQTTConfig(broker="mqtt.local")
        connector = HomeAssistantMQTTConnector(config)
        assert connector._password is None

    def test_trailing_slash_stripped_from_prefix(self):
        """Test that trailing slash is stripped from topic_prefix."""
        config = HomeAssistantMQTTConfig(broker="mqtt.local", topic_prefix="ha/")
        connector = HomeAssistantMQTTConnector(config)
        assert connector._topic_prefix == "ha"

    def test_warns_missing_broker(self):
        """Test warning when broker is missing."""
        with patch(
            "actions.home_assistant.connector.mqtt.logging.warning"
        ) as mock_warn:
            config = HomeAssistantMQTTConfig(broker="")
            HomeAssistantMQTTConnector(config)
            assert any("broker" in str(c) for c in mock_warn.call_args_list)


class TestBuildTopic:
    """Tests for _build_topic()."""

    def test_build_topic_with_dot_entity_id(self):
        """Test topic building with standard entity_id format."""
        connector = make_connector()
        topic = connector._build_topic("light", "light.living_room")
        assert topic == "homeassistant/light/living_room/set"

    def test_build_topic_without_dot(self):
        """Test topic building when entity_id has no dot."""
        connector = make_connector()
        topic = connector._build_topic("switch", "fan")
        assert topic == "homeassistant/switch/fan/set"

    def test_build_topic_custom_prefix(self):
        """Test topic building with custom prefix."""
        config = HomeAssistantMQTTConfig(broker="mqtt.local", topic_prefix="myhome")
        connector = HomeAssistantMQTTConnector(config)
        topic = connector._build_topic("light", "light.bedroom")
        assert topic == "myhome/light/bedroom/set"


class TestPublish:
    """Tests for _publish()."""

    @pytest.mark.asyncio
    async def test_publish_success(self):
        """Test successful publish."""
        connector = make_connector()
        ctx, mock_client = mock_mqtt_session()
        with ctx:
            result = await connector._publish(
                "homeassistant/light/x/set", {"state": "ON"}
            )
            assert result is True
            mock_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_no_broker(self):
        """Test that missing broker returns False."""
        config = HomeAssistantMQTTConfig(broker="")
        connector = HomeAssistantMQTTConnector(config)
        result = await connector._publish("topic", {"state": "ON"})
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_mqtt_error(self):
        """Test handling of MqttError."""
        import aiomqtt

        connector = make_connector()
        with patch("actions.home_assistant.connector.mqtt.aiomqtt.Client") as mock_cls:
            mock_cls.side_effect = aiomqtt.MqttError("connection refused")
            with patch(
                "actions.home_assistant.connector.mqtt.logging.error"
            ) as mock_err:
                result = await connector._publish("topic", {"state": "ON"})
                assert result is False
                assert any("MQTT error" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_publish_timeout(self):
        """Test handling of timeout error."""
        connector = make_connector()
        with patch("actions.home_assistant.connector.mqtt.aiomqtt.Client") as mock_cls:
            mock_cls.side_effect = TimeoutError()
            with patch(
                "actions.home_assistant.connector.mqtt.logging.error"
            ) as mock_err:
                result = await connector._publish("topic", {"state": "ON"})
                assert result is False
                assert any("timed out" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_publish_unexpected_exception(self):
        """Test handling of unexpected exception."""
        connector = make_connector()
        with patch("actions.home_assistant.connector.mqtt.aiomqtt.Client") as mock_cls:
            mock_cls.side_effect = RuntimeError("boom")
            with patch(
                "actions.home_assistant.connector.mqtt.logging.error"
            ) as mock_err:
                result = await connector._publish("topic", {"state": "ON"})
                assert result is False
                assert any(
                    "unexpected error" in str(c) for c in mock_err.call_args_list
                )


class TestConnectLight:
    """Tests for light device control."""

    @pytest.mark.asyncio
    async def test_light_turn_on(self):
        """Test light turn on."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.TURN_ON,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/light/living_room/set", {"state": "ON"}
            )

    @pytest.mark.asyncio
    async def test_light_turn_off(self):
        """Test light turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.TURN_OFF,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/light/living_room/set", {"state": "OFF"}
            )

    @pytest.mark.asyncio
    async def test_light_set_brightness(self):
        """Test light set brightness."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.SET_BRIGHTNESS,
            brightness=200,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/light/living_room/set",
                {"state": "ON", "brightness": 200},
            )

    @pytest.mark.asyncio
    async def test_light_set_color(self):
        """Test light set color."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.LIGHT,
            entity_id="light.living_room",
            action=HAAction.SET_COLOR,
            color="blue",
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/light/living_room/set",
                {"state": "ON", "hs_color": COLOR_MAP["blue"]},
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
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/light/x/set",
                {"state": "ON", "hs_color": COLOR_MAP["white"]},
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
            "actions.home_assistant.connector.mqtt.logging.warning"
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
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/switch/fan/set", {"state": "ON"}
            )

    @pytest.mark.asyncio
    async def test_switch_turn_off(self):
        """Test switch turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.SWITCH,
            entity_id="switch.fan",
            action=HAAction.TURN_OFF,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/switch/fan/set", {"state": "OFF"}
            )

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
            "actions.home_assistant.connector.mqtt.logging.warning"
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
            temperature=22.0,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/climate/bedroom/set", {"temperature": 22.0}
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
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/climate/bedroom/set", {"state": "ON"}
            )

    @pytest.mark.asyncio
    async def test_climate_turn_off(self):
        """Test climate turn off."""
        connector = make_connector()
        inp = HomeAssistantInput(
            device_type=HADeviceType.CLIMATE,
            entity_id="climate.bedroom",
            action=HAAction.TURN_OFF,
        )
        with patch.object(connector, "_publish", new_callable=AsyncMock) as mock_pub:
            await connector.connect(inp)
            mock_pub.assert_called_once_with(
                "homeassistant/climate/bedroom/set", {"state": "OFF"}
            )

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
            "actions.home_assistant.connector.mqtt.logging.warning"
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
            "actions.home_assistant.connector.mqtt.logging.warning"
        ) as mock_warn:
            with patch(
                "actions.home_assistant.connector.mqtt.HADeviceType"
            ) as mock_enum:
                mock_enum.LIGHT = "FAKE"
                mock_enum.SWITCH = "FAKE"
                mock_enum.CLIMATE = "FAKE"
                await connector.connect(inp)
                assert any("not supported" in str(c) for c in mock_warn.call_args_list)
