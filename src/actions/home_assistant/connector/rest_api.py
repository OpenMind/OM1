import asyncio
import logging
from typing import Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    COLOR_MAP,
    HAAction,
    HADeviceType,
    HomeAssistantInput,
)


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for Home Assistant REST API connector.

    Parameters
    ----------
    base_url : str
        Home Assistant base URL (e.g. http://homeassistant.local:8123).
    token : str
        Long-lived access token from Home Assistant profile.
    timeout : float
        Request timeout in seconds.
    """

    base_url: str = Field(default="", description="Home Assistant base URL")
    token: str = Field(default="", description="Long-lived access token")
    timeout: float = Field(default=10.0, description="Request timeout in seconds")


class HomeAssistantRESTConnector(
    ActionConnector[HomeAssistantConfig, HomeAssistantInput]
):
    """
    Connector for Home Assistant REST API.

    Controls smart home devices via the Home Assistant REST API.
    Supports lights, switches, and climate devices.
    """

    def __init__(self, config: HomeAssistantConfig):
        """
        Initialize the Home Assistant REST connector.

        Parameters
        ----------
        config : HomeAssistantConfig
            Configuration for the connector.
        """
        super().__init__(config)

        if not self.config.base_url:
            logging.warning(
                "HomeAssistantRESTConnector: base_url not provided in configuration"
            )
        if not self.config.token:
            logging.warning(
                "HomeAssistantRESTConnector: token not provided in configuration"
            )

    def _get_headers(self) -> dict:
        """Build authorization headers."""
        return {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
        }

    def _build_service_url(self, domain: str, service: str) -> str:
        """Build the HA service call URL."""
        base = self.config.base_url.rstrip("/")
        return f"{base}/api/services/{domain}/{service}"

    async def _call_service(
        self, domain: str, service: str, payload: dict
    ) -> Optional[dict]:
        """
        Call a Home Assistant service via REST API.

        Parameters
        ----------
        domain : str
            HA domain (e.g. light, switch, climate).
        service : str
            HA service (e.g. turn_on, turn_off).
        payload : dict
            Service call payload.

        Returns
        -------
        Optional[dict]
            Response data or None on failure.
        """
        if not self.config.base_url or not self.config.token:
            logging.error("HomeAssistantRESTConnector: base_url or token not set")
            return None

        url = self._build_service_url(domain, service)
        logging.info(f"HomeAssistantRESTConnector: POST {url} payload={payload}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, headers=self._get_headers(), json=payload
                ) as response:
                    if response.status in (200, 201):
                        logging.info(
                            f"HomeAssistantRESTConnector: success {response.status}"
                        )
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"HomeAssistantRESTConnector: error {response.status} - {error_text}"
                        )
                        return None
        except asyncio.TimeoutError:
            logging.error("HomeAssistantRESTConnector: request timed out")
            return None
        except aiohttp.ClientError as e:
            logging.error(f"HomeAssistantRESTConnector: network error - {e}")
            return None
        except Exception as e:
            logging.error(f"HomeAssistantRESTConnector: unexpected error - {e}")
            return None

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute a Home Assistant device control action.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            Input containing device type, entity_id, action, and parameters.
        """
        if not output_interface.entity_id:
            logging.warning("HomeAssistantRESTConnector: entity_id is empty, skipping")
            return

        entity_id = output_interface.entity_id
        action = output_interface.action
        device_type = output_interface.device_type

        logging.info(
            f"HomeAssistantRESTConnector: {action.value} on {entity_id} "
            f"(type={device_type.value})"
        )

        # LIGHT
        if device_type == HADeviceType.LIGHT:
            if action == HAAction.TURN_ON:
                await self._call_service("light", "turn_on", {"entity_id": entity_id})
            elif action == HAAction.TURN_OFF:
                await self._call_service("light", "turn_off", {"entity_id": entity_id})
            elif action == HAAction.SET_BRIGHTNESS:
                brightness = max(0, min(255, output_interface.brightness))
                await self._call_service(
                    "light",
                    "turn_on",
                    {"entity_id": entity_id, "brightness": brightness},
                )
            elif action == HAAction.SET_COLOR:
                color_name = output_interface.color.lower()
                hs_color = COLOR_MAP.get(color_name, COLOR_MAP["white"])
                await self._call_service(
                    "light",
                    "turn_on",
                    {"entity_id": entity_id, "hs_color": hs_color},
                )
            else:
                logging.warning(
                    f"HomeAssistantRESTConnector: action '{action.value}' "
                    f"not supported for light"
                )

        # SWITCH
        elif device_type == HADeviceType.SWITCH:
            if action == HAAction.TURN_ON:
                await self._call_service("switch", "turn_on", {"entity_id": entity_id})
            elif action == HAAction.TURN_OFF:
                await self._call_service("switch", "turn_off", {"entity_id": entity_id})
            else:
                logging.warning(
                    f"HomeAssistantRESTConnector: action '{action.value}' "
                    f"not supported for switch"
                )

        # CLIMATE
        elif device_type == HADeviceType.CLIMATE:
            if action == HAAction.SET_TEMPERATURE:
                await self._call_service(
                    "climate",
                    "set_temperature",
                    {
                        "entity_id": entity_id,
                        "temperature": output_interface.temperature,
                    },
                )
            elif action == HAAction.TURN_ON:
                await self._call_service("climate", "turn_on", {"entity_id": entity_id})
            elif action == HAAction.TURN_OFF:
                await self._call_service(
                    "climate", "turn_off", {"entity_id": entity_id}
                )
            else:
                logging.warning(
                    f"HomeAssistantRESTConnector: action '{action.value}' "
                    f"not supported for climate"
                )

        else:
            logging.warning(
                f"HomeAssistantRESTConnector: device_type '{device_type.value}' "
                f"not supported"
            )
