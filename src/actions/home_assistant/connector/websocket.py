"""Home Assistant WebSocket API action connector."""

import json
import logging
from typing import Optional

import websockets
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    COLOR_MAP,
    HAAction,
    HADeviceType,
    HomeAssistantInput,
)


class HomeAssistantWebSocketConfig(ActionConfig):
    """
    Configuration for Home Assistant WebSocket connector.

    Parameters
    ----------
    base_url : str
        Home Assistant base URL (e.g. http://homeassistant.local:8123).
    token : str
        Long-lived access token from Home Assistant profile.
    timeout : float
        Connection and command timeout in seconds.
    """

    base_url: str = Field(default="", description="Home Assistant base URL")
    token: str = Field(default="", description="Long-lived access token")
    timeout: float = Field(default=10.0, description="Timeout in seconds")


class HomeAssistantWebSocketConnector(
    ActionConnector[HomeAssistantWebSocketConfig, HomeAssistantInput]
):
    """
    Home Assistant connector using WebSocket API.

    Provides persistent bidirectional connection to Home Assistant,
    enabling real-time command execution with result confirmation.
    """

    def __init__(self, config: HomeAssistantWebSocketConfig):
        """
        Initialize the WebSocket connector.

        Parameters
        ----------
        config : HomeAssistantWebSocketConfig
            Configuration for the connector.
        """
        super().__init__(config)
        base = config.base_url.rstrip("/")
        if base.startswith("https://"):
            self._ws_url = "wss://" + base[len("https://") :] + "/api/websocket"
        elif base.startswith("http://"):
            self._ws_url = "ws://" + base[len("http://") :] + "/api/websocket"
        else:
            self._ws_url = base + "/api/websocket"
        self._token = config.token
        self._timeout = config.timeout
        self._msg_id = 1

        if not config.base_url:
            logging.warning("HomeAssistantWebSocketConnector: base_url not provided")
        if not config.token:
            logging.warning("HomeAssistantWebSocketConnector: token not provided")

    async def _send_command(
        self,
        domain: str,
        service: str,
        entity_id: str,
        service_data: Optional[dict] = None,
    ) -> bool:
        """
        Open a WebSocket connection, authenticate, and send a service command.

        Parameters
        ----------
        domain : str
            Service domain (e.g. 'light', 'switch', 'climate').
        service : str
            Service name (e.g. 'turn_on', 'turn_off').
        entity_id : str
            Target entity ID.
        service_data : Optional[dict]
            Additional service data (brightness, temperature, etc.).

        Returns
        -------
        bool
            True if command succeeded, False otherwise.
        """
        if not self._ws_url or not self._token:
            return False

        try:
            async with websockets.connect(
                self._ws_url, open_timeout=self._timeout
            ) as ws:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") != "auth_required":
                    logging.error(
                        f"HomeAssistantWebSocketConnector: expected auth_required, "
                        f"got {msg.get('type')}"
                    )
                    return False

                await ws.send(json.dumps({"type": "auth", "access_token": self._token}))
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") != "auth_ok":
                    logging.error(
                        "HomeAssistantWebSocketConnector: authentication failed"
                    )
                    return False

                command = {
                    "id": self._msg_id,
                    "type": "call_service",
                    "domain": domain,
                    "service": service,
                    "target": {"entity_id": entity_id},
                    "service_data": service_data or {},
                }
                self._msg_id += 1

                await ws.send(json.dumps(command))
                raw = await ws.recv()
                result = json.loads(raw)

                if result.get("success"):
                    return True
                else:
                    error = result.get("error", {})
                    logging.error(
                        f"HomeAssistantWebSocketConnector: command failed - "
                        f"{error.get('code')}: {error.get('message')}"
                    )
                    return False

        except TimeoutError:
            logging.error("HomeAssistantWebSocketConnector: connection timed out")
            return False
        except websockets.exceptions.WebSocketException as e:
            logging.error(f"HomeAssistantWebSocketConnector: WebSocket error - {e}")
            return False
        except Exception as e:
            logging.error(f"HomeAssistantWebSocketConnector: unexpected error - {e}")
            return False

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute a Home Assistant device control action via WebSocket.

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
            if action == HAAction.TURN_ON:
                await self._send_command("light", "turn_on", entity_id)
            elif action == HAAction.TURN_OFF:
                await self._send_command("light", "turn_off", entity_id)
            elif action == HAAction.SET_BRIGHTNESS:
                brightness = output_interface.brightness or 255
                await self._send_command(
                    "light", "turn_on", entity_id, {"brightness": brightness}
                )
            elif action == HAAction.SET_COLOR:
                color = (output_interface.color or "white").lower()
                hs_color = COLOR_MAP.get(color, COLOR_MAP["white"])
                await self._send_command(
                    "light", "turn_on", entity_id, {"hs_color": hs_color}
                )
            else:
                logging.warning(
                    f"HomeAssistantWebSocketConnector: action '{action.value}' "
                    f"not supported for light"
                )

        elif device_type == HADeviceType.SWITCH:
            if action == HAAction.TURN_ON:
                await self._send_command("switch", "turn_on", entity_id)
            elif action == HAAction.TURN_OFF:
                await self._send_command("switch", "turn_off", entity_id)
            else:
                logging.warning(
                    f"HomeAssistantWebSocketConnector: action '{action.value}' "
                    f"not supported for switch"
                )

        elif device_type == HADeviceType.CLIMATE:
            if action == HAAction.SET_TEMPERATURE:
                temperature = output_interface.temperature or 20.0
                await self._send_command(
                    "climate",
                    "set_temperature",
                    entity_id,
                    {"temperature": temperature},
                )
            elif action == HAAction.TURN_ON:
                await self._send_command("climate", "turn_on", entity_id)
            elif action == HAAction.TURN_OFF:
                await self._send_command("climate", "turn_off", entity_id)
            else:
                logging.warning(
                    f"HomeAssistantWebSocketConnector: action '{action.value}' "
                    f"not supported for climate"
                )

        else:
            logging.warning(
                f"HomeAssistantWebSocketConnector: device_type '{device_type.value}' "
                f"not supported"
            )
