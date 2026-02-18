import logging
from typing import Any, Optional

import aiohttp

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    HomeAssistantAction,
    HomeAssistantDeviceType,
    HomeAssistantInput,
)


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for Home Assistant connector.

    Parameters
    ----------
    base_url : str
        Base URL of Home Assistant instance (e.g., "http://homeassistant.local:8123")
    token : str
        Long-lived access token for Home Assistant API
    """

    base_url: str = "http://homeassistant.local:8123"
    token: str = ""


class HomeAssistantConnector(ActionConnector[HomeAssistantConfig, HomeAssistantInput]):
    """
    Connector to link Home Assistant actions with Home Assistant API.

    This connector communicates with Home Assistant via REST API to control
    IoT devices such as lights, switches, and thermostats.
    """

    def __init__(self, config: HomeAssistantConfig):
        """
        Initialize the HomeAssistantConnector with the given configuration.

        Parameters
        ----------
        config : HomeAssistantConfig
            Configuration parameters including base_url and access token.
        """
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.

        Returns
        -------
        aiohttp.ClientSession
            The HTTP session for API calls.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers with authorization token.

        Returns
        -------
        dict[str, str]
            Headers dict with Authorization bearer token.
        """
        return {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
        }

    async def _call_service(
        self, domain: str, service: str, service_data: dict[str, Any]
    ) -> bool:
        """
        Call a Home Assistant service.

        Parameters
        ----------
        domain : str
            Service domain (e.g., "light", "switch", "climate")
        service : str
            Service name (e.g., "turn_on", "turn_off")
        service_data : dict[str, Any]
            Additional data for the service call

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        url = f"{self.config.base_url}/api/services/{domain}/{service}"

        try:
            session = await self._get_session()
            async with session.post(
                url, headers=self._get_headers(), json=service_data
            ) as response:
                if response.status == 200:
                    logging.info(f"Home Assistant service {domain}.{service} called successfully")
                    return True
                else:
                    error_text = await response.text()
                    logging.error(
                        f"Home Assistant API error: {response.status} - {error_text}"
                    )
                    return False
        except Exception as e:
            logging.error(f"Failed to call Home Assistant service: {e}")
            return False

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Connect to Home Assistant and execute the requested action.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The input containing device info and action to perform.
        """
        device_id = output_interface.device_id
        action = output_interface.action
        device_type = output_interface.device_type

        logging.info(f"Home Assistant: {action} on {device_id}")

        # Map device type to Home Assistant domain
        domain = device_type.value

        if action == HomeAssistantAction.TURN_ON:
            service_data: dict[str, Any] = {"entity_id": device_id}

            # Add brightness if specified for lights
            if (
                output_interface.brightness is not None
                and device_type == HomeAssistantDeviceType.LIGHT
            ):
                service_data["brightness"] = output_interface.brightness

            # Add color if specified for lights
            if (
                output_interface.color is not None
                and device_type == HomeAssistantDeviceType.LIGHT
            ):
                service_data["color_name"] = output_interface.color

            await self._call_service(domain, "turn_on", service_data)

        elif action == HomeAssistantAction.TURN_OFF:
            await self._call_service(domain, "turn_off", {"entity_id": device_id})

        elif action == HomeAssistantAction.SET_BRIGHTNESS:
            if output_interface.brightness is not None:
                await self._call_service(
                    domain,
                    "turn_on",
                    {"entity_id": device_id, "brightness": output_interface.brightness},
                )
            else:
                logging.warning("SET_BRIGHTNESS called without brightness value")

        elif action == HomeAssistantAction.SET_COLOR:
            if output_interface.color is not None:
                await self._call_service(
                    domain,
                    "turn_on",
                    {"entity_id": device_id, "color_name": output_interface.color},
                )
            else:
                logging.warning("SET_COLOR called without color value")

        elif action == HomeAssistantAction.SET_TEMPERATURE:
            if (
                output_interface.temperature is not None
                and device_type == HomeAssistantDeviceType.THERMOSTAT
            ):
                await self._call_service(
                    domain,
                    "set_temperature",
                    {
                        "entity_id": device_id,
                        "temperature": output_interface.temperature,
                    },
                )
            else:
                logging.warning(
                    "SET_TEMPERATURE called without temperature value or wrong device type"
                )

        else:
            logging.warning(f"Unknown Home Assistant action: {action}")

    async def stop(self) -> None:
        """
        Stop the connector and close the HTTP session.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
