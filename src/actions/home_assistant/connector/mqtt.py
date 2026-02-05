"""
Home Assistant MQTT Connector for OM1.

This module provides a connector to control Home Assistant devices using
MQTT protocol. It supports lights, switches, and other smart home devices
that are exposed via MQTT in Home Assistant.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    HomeAssistantInput,
    HomeAssistantOutput,
)

try:
    import aiomqtt

    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning(
        "aiomqtt not installed. MQTT connector will not be available. "
        "Install with: pip install aiomqtt"
    )


class HomeAssistantMQTTConfig(ActionConfig):
    """
    Configuration for Home Assistant MQTT connector.

    Parameters
    ----------
    mqtt_host : str
        MQTT broker hostname.
    mqtt_port : int
        MQTT broker port.
    mqtt_username : Optional[str]
        MQTT username for authentication.
    mqtt_password : Optional[str]
        MQTT password for authentication.
    topic_prefix : str
        Home Assistant MQTT topic prefix (default: 'homeassistant').
    """

    mqtt_host: str = Field(
        default="localhost",
        description="MQTT broker hostname",
    )
    mqtt_port: int = Field(
        default=1883,
        description="MQTT broker port",
    )
    mqtt_username: Optional[str] = Field(
        default=None,
        description="MQTT username for authentication",
    )
    mqtt_password: Optional[str] = Field(
        default=None,
        description="MQTT password for authentication",
    )
    topic_prefix: str = Field(
        default="homeassistant",
        description="Home Assistant MQTT topic prefix",
    )


class HomeAssistantMQTTConnector(
    ActionConnector[HomeAssistantMQTTConfig, HomeAssistantInput]
):
    """
    A connector for controlling Home Assistant devices via MQTT.

    This connector uses the MQTT protocol to communicate with Home Assistant
    devices. It requires an MQTT broker (like Mosquitto) and Home Assistant
    MQTT integration to be configured.

    Supports:
    - Lights: on/off, brightness, color (RGB)
    - Switches: on/off
    - Thermostats: temperature setting
    """

    def __init__(self, config: HomeAssistantMQTTConfig):
        """
        Initialize the Home Assistant MQTT connector.

        Parameters
        ----------
        config : HomeAssistantMQTTConfig
            Configuration for the connector.
        """
        super().__init__(config)

        if not MQTT_AVAILABLE:
            raise ImportError(
                "aiomqtt is required for MQTT connector. "
                "Install with: pip install aiomqtt"
            )

        self.mqtt_host = config.mqtt_host
        self.mqtt_port = config.mqtt_port
        self.mqtt_username = config.mqtt_username
        self.mqtt_password = config.mqtt_password
        self.topic_prefix = config.topic_prefix
        self._last_result: Optional[HomeAssistantOutput] = None
        self._client: Optional[aiomqtt.Client] = None

        logging.info(
            f"HomeAssistant MQTT connector initialized for {self.mqtt_host}:{self.mqtt_port}"
        )

    def _get_entity_topic(self, entity_id: str, action_type: str = "set") -> str:
        """
        Generate MQTT topic for an entity.

        Uses the Home Assistant MQTT discovery topic format:
        ``{topic_prefix}/{domain}/{object_id}/{action_type}``

        For example, entity_id ``light.living_room`` with action_type ``set``
        becomes ``homeassistant/light/living_room/set``.

        Parameters
        ----------
        entity_id : str
            Home Assistant entity ID (e.g., 'light.living_room').
        action_type : str
            Topic type: 'set' for commands, 'state' for state.

        Returns
        -------
        str
            MQTT topic for the entity.
        """
        # Convert entity_id (e.g., 'light.living_room') to MQTT topic
        domain, name = entity_id.split(".", 1)
        return f"{self.topic_prefix}/{domain}/{name}/{action_type}"

    async def _get_client(self) -> aiomqtt.Client:
        """
        Get or create a persistent MQTT client connection.

        Returns
        -------
        aiomqtt.Client
            Connected MQTT client.
        """
        if self._client is None:
            self._client = aiomqtt.Client(
                hostname=self.mqtt_host,
                port=self.mqtt_port,
                username=self.mqtt_username,
                password=self.mqtt_password,
            )
            await self._client.__aenter__()
            logging.info(
                f"MQTT persistent connection established to "
                f"{self.mqtt_host}:{self.mqtt_port}"
            )
        return self._client

    async def _publish_message(
        self,
        topic: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Publish a message to MQTT broker using a persistent connection.

        Parameters
        ----------
        topic : str
            MQTT topic to publish to.
        payload : Dict[str, Any]
            Message payload as dictionary.

        Returns
        -------
        bool
            True if publish was successful.
        """
        try:
            client = await self._get_client()
            await client.publish(
                topic,
                payload=json.dumps(payload),
                qos=1,
            )
            logging.info(f"Published to {topic}: {payload}")
            return True
        except Exception as e:
            logging.error(f"MQTT publish error: {e}")
            # Reset client on error so next call reconnects
            self._client = None
            return False

    async def _control_light(
        self,
        entity_id: str,
        action: str,
        brightness: Optional[int] = None,
        color_rgb: Optional[str] = None,
    ) -> HomeAssistantOutput:
        """Control a light device via MQTT."""
        topic = self._get_entity_topic(entity_id, "set")
        payload: Dict[str, Any] = {}

        if action in ("turn_on", "on"):
            payload["state"] = "ON"
            if brightness is not None:
                payload["brightness"] = max(0, min(255, brightness))
            if color_rgb:
                try:
                    rgb = [int(x.strip()) for x in color_rgb.split(",")]
                    if len(rgb) == 3:
                        if all(0 <= v <= 255 for v in rgb):
                            payload["color"] = {"r": rgb[0], "g": rgb[1], "b": rgb[2]}
                        else:
                            logging.warning(
                                f"RGB values out of range (0-255): {color_rgb}"
                            )
                except ValueError:
                    logging.warning(f"Invalid RGB color format: {color_rgb}")
        elif action in ("turn_off", "off"):
            payload["state"] = "OFF"
        elif action == "brightness" and brightness is not None:
            payload["state"] = "ON"
            payload["brightness"] = max(0, min(255, brightness))
        elif action == "color" and color_rgb:
            payload["state"] = "ON"
            try:
                rgb = [int(x.strip()) for x in color_rgb.split(",")]
                if len(rgb) == 3:
                    if all(0 <= v <= 255 for v in rgb):
                        payload["color"] = {"r": rgb[0], "g": rgb[1], "b": rgb[2]}
                    else:
                        return HomeAssistantOutput(
                            success=False,
                            message=f"RGB values out of range (0-255): {color_rgb}",
                            entity_id=entity_id,
                        )
            except ValueError:
                return HomeAssistantOutput(
                    success=False,
                    message=f"Invalid RGB color format: {color_rgb}",
                    entity_id=entity_id,
                )
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown light action: {action}",
                entity_id=entity_id,
            )

        success = await self._publish_message(topic, payload)

        return HomeAssistantOutput(
            success=success,
            message=f"Light {action} {'successful' if success else 'failed'}",
            entity_id=entity_id,
            new_state=payload.get("state"),
        )

    async def _control_switch(
        self,
        entity_id: str,
        action: str,
    ) -> HomeAssistantOutput:
        """Control a switch device via MQTT."""
        topic = self._get_entity_topic(entity_id, "set")

        if action in ("turn_on", "on"):
            payload = {"state": "ON"}
        elif action in ("turn_off", "off"):
            payload = {"state": "OFF"}
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown switch action: {action}",
                entity_id=entity_id,
            )

        success = await self._publish_message(topic, payload)

        return HomeAssistantOutput(
            success=success,
            message=f"Switch {action} {'successful' if success else 'failed'}",
            entity_id=entity_id,
            new_state=payload.get("state"),
        )

    async def _control_thermostat(
        self,
        entity_id: str,
        action: str,
        temperature: Optional[float] = None,
        hvac_mode: Optional[str] = None,
    ) -> HomeAssistantOutput:
        """Control a climate/thermostat device via MQTT."""
        payload: Dict[str, Any] = {}

        if action == "set_temperature" and temperature is not None:
            topic = self._get_entity_topic(entity_id, "temperature_command")
            payload["temperature"] = temperature
        elif action == "set_hvac_mode" and hvac_mode:
            topic = self._get_entity_topic(entity_id, "mode_command")
            payload["mode"] = hvac_mode
        elif action in ("turn_off", "off"):
            topic = self._get_entity_topic(entity_id, "mode_command")
            payload["mode"] = "off"
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown thermostat action: {action}",
                entity_id=entity_id,
            )

        success = await self._publish_message(topic, payload)

        return HomeAssistantOutput(
            success=success,
            message=f"Thermostat {action} {'successful' if success else 'failed'}",
            entity_id=entity_id,
        )

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Connect and execute the Home Assistant action via MQTT.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The input containing the device type, entity ID, and action.
        """
        device_type = output_interface.device_type.lower()
        entity_id = output_interface.entity_id
        action = output_interface.action.lower()

        logging.info(
            f"HomeAssistant MQTT action: {device_type}/{action} on {entity_id}"
        )

        # Route to appropriate device handler
        if device_type in ("light", "lights"):
            result = await self._control_light(
                entity_id=entity_id,
                action=action,
                brightness=output_interface.brightness,
                color_rgb=output_interface.color_rgb,
            )
        elif device_type in ("switch", "switches"):
            result = await self._control_switch(
                entity_id=entity_id,
                action=action,
            )
        elif device_type in ("climate", "thermostat", "thermostats"):
            result = await self._control_thermostat(
                entity_id=entity_id,
                action=action,
                temperature=output_interface.temperature,
                hvac_mode=output_interface.hvac_mode,
            )
        else:
            result = HomeAssistantOutput(
                success=False,
                message=f"Unsupported device type for MQTT: {device_type}",
                entity_id=entity_id,
            )

        self._last_result = result

        if result.success:
            logging.info(f"HomeAssistant MQTT: {result.message}")
        else:
            logging.error(f"HomeAssistant MQTT error: {result.message}")

    def tick(self) -> None:
        """
        Tick method for periodic updates.
        """
        time.sleep(60)

    async def close(self) -> None:
        """Close the persistent MQTT client connection."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
                logging.info("MQTT persistent connection closed")
            except Exception as e:
                logging.error(f"Error closing MQTT connection: {e}")
            finally:
                self._client = None