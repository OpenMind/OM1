"""Home Assistant MQTT action connector."""

import json
import logging

import aiomqtt
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    COLOR_MAP,
    HAAction,
    HADeviceType,
    HomeAssistantInput,
)


class HomeAssistantMQTTConfig(ActionConfig):
    """
    Configuration for Home Assistant MQTT connector.

    Parameters
    ----------
    broker : str
        MQTT broker hostname or IP address.
    port : int
        MQTT broker port (default: 1883).
    username : str
        MQTT broker username (optional).
    password : str
        MQTT broker password (optional).
    topic_prefix : str
        Base topic prefix for Home Assistant (default: homeassistant).
    timeout : float
        Connection timeout in seconds.
    """

    broker: str = Field(default="", description="MQTT broker hostname or IP")
    port: int = Field(default=1883, description="MQTT broker port")
    username: str = Field(default="", description="MQTT broker username")
    password: str = Field(default="", description="MQTT broker password")
    topic_prefix: str = Field(default="homeassistant", description="Base topic prefix")
    timeout: float = Field(default=10.0, description="Connection timeout in seconds")


class HomeAssistantMQTTConnector(
    ActionConnector[HomeAssistantMQTTConfig, HomeAssistantInput]
):
    """
    Home Assistant connector using MQTT protocol.

    Publishes device control commands to MQTT topics following
    the Home Assistant MQTT convention:
    <prefix>/<domain>/<entity>/set
    """

    def __init__(self, config: HomeAssistantMQTTConfig):
        """
        Initialize the MQTT connector.

        Parameters
        ----------
        config : HomeAssistantMQTTConfig
            Configuration for the connector.
        """
        super().__init__(config)
        self._broker = config.broker
        self._port = config.port
        self._username = config.username or None
        self._password = config.password or None
        self._topic_prefix = config.topic_prefix.rstrip("/")
        self._timeout = config.timeout

        if not config.broker:
            logging.warning("HomeAssistantMQTTConnector: broker not provided")

    def _build_topic(self, domain: str, entity_id: str) -> str:
        """
        Build MQTT command topic for an entity.

        Parameters
        ----------
        domain : str
            Device domain (e.g. 'light', 'switch', 'climate').
        entity_id : str
            Full entity ID (e.g. 'light.living_room').

        Returns
        -------
        str
            MQTT topic string.
        """
        name = entity_id.split(".", 1)[-1] if "." in entity_id else entity_id
        return f"{self._topic_prefix}/{domain}/{name}/set"

    async def _publish(self, topic: str, payload: dict) -> bool:
        """
        Publish a JSON payload to an MQTT topic.

        Parameters
        ----------
        topic : str
            MQTT topic to publish to.
        payload : dict
            Payload to serialize as JSON.

        Returns
        -------
        bool
            True if published successfully, False otherwise.
        """
        if not self._broker:
            return False

        try:
            async with aiomqtt.Client(
                hostname=self._broker,
                port=self._port,
                username=self._username,
                password=self._password,
                timeout=self._timeout,
            ) as client:
                await client.publish(topic, json.dumps(payload))
                return True
        except aiomqtt.MqttError as e:
            logging.error(f"HomeAssistantMQTTConnector: MQTT error - {e}")
            return False
        except TimeoutError:
            logging.error("HomeAssistantMQTTConnector: connection timed out")
            return False
        except Exception as e:
            logging.error(f"HomeAssistantMQTTConnector: unexpected error - {e}")
            return False

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute a Home Assistant device control action via MQTT.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The action request containing device_type, entity_id, action,
            and optional parameters.
        """
        device_type = output_interface.device_type
        entity_id = output_interface.entity_id
        action = output_interface.action

        if device_type == HADeviceType.LIGHT:
            topic = self._build_topic("light", entity_id)
            if action == HAAction.TURN_ON:
                await self._publish(topic, {"state": "ON"})
            elif action == HAAction.TURN_OFF:
                await self._publish(topic, {"state": "OFF"})
            elif action == HAAction.SET_BRIGHTNESS:
                brightness = output_interface.brightness or 255
                await self._publish(topic, {"state": "ON", "brightness": brightness})
            elif action == HAAction.SET_COLOR:
                color = (output_interface.color or "white").lower()
                hs_color = COLOR_MAP.get(color, COLOR_MAP["white"])
                await self._publish(topic, {"state": "ON", "hs_color": hs_color})
            else:
                logging.warning(
                    f"HomeAssistantMQTTConnector: action '{action.value}' "
                    f"not supported for light"
                )

        elif device_type == HADeviceType.SWITCH:
            topic = self._build_topic("switch", entity_id)
            if action == HAAction.TURN_ON:
                await self._publish(topic, {"state": "ON"})
            elif action == HAAction.TURN_OFF:
                await self._publish(topic, {"state": "OFF"})
            else:
                logging.warning(
                    f"HomeAssistantMQTTConnector: action '{action.value}' "
                    f"not supported for switch"
                )

        elif device_type == HADeviceType.CLIMATE:
            topic = self._build_topic("climate", entity_id)
            if action == HAAction.SET_TEMPERATURE:
                temperature = output_interface.temperature or 20.0
                await self._publish(topic, {"temperature": temperature})
            elif action == HAAction.TURN_ON:
                await self._publish(topic, {"state": "ON"})
            elif action == HAAction.TURN_OFF:
                await self._publish(topic, {"state": "OFF"})
            else:
                logging.warning(
                    f"HomeAssistantMQTTConnector: action '{action.value}' "
                    f"not supported for climate"
                )

        else:
            logging.warning(
                f"HomeAssistantMQTTConnector: device_type '{device_type.value}' "
                f"not supported"
            )
