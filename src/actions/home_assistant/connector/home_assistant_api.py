import logging
from typing import Any, Dict, Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantInput, HomeAssistantOutput


class HomeAssistantAPIConfig(ActionConfig):
    """
    Configuration class for HomeAssistantAPIConnector.

    Attributes
    ----------
    base_url : str
        Home Assistant base URL (e.g., "http://192.168.1.100:8123")
    access_token : str
        Long-lived access token for Home Assistant API authentication
    verify_ssl : bool
        Whether to verify SSL certificates (default: True)
    timeout : int
        Request timeout in seconds (default: 10)
    """

    base_url: str = Field(
        description="Home Assistant base URL (e.g., http://localhost:8123)"
    )
    access_token: str = Field(
        description="Long-lived access token for Home Assistant API"
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    timeout: int = Field(default=10, description="Request timeout in seconds")


class HomeAssistantAPIConnector(ActionConnector[HomeAssistantAPIConfig, HomeAssistantInput]):
    """
    Connector for Home Assistant REST API.

    This connector integrates with Home Assistant to control smart home devices
    including lights, switches, and thermostats via the REST API.

    Supported operations:
    - turn_on: Turn on a device
    - turn_off: Turn off a device
    - set_brightness: Set light brightness (0-255)
    - set_color: Set light RGB color
    - set_temperature: Set thermostat target temperature
    - get_state: Query current device state
    """

    def __init__(self, config: HomeAssistantAPIConfig):
        """
        Initialize the Home Assistant API connector.

        Parameters
        ----------
        config : HomeAssistantAPIConfig
            Configuration object for the connector.
        """
        super().__init__(config)
        self._headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

        if not self.config.base_url:
            logging.warning("Home Assistant base URL not provided in configuration")
        if not self.config.access_token:
            logging.warning("Home Assistant access token not provided in configuration")

    def _get_api_url(self, endpoint: str) -> str:
        """
        Construct full API URL from endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint path

        Returns
        -------
        str
            Full API URL
        """
        base = self.config.base_url.rstrip("/")
        return f"{base}/api/{endpoint.lstrip('/')}"

    def _get_domain_from_entity(self, entity_id: str) -> str:
        """
        Extract domain from entity_id.

        Parameters
        ----------
        entity_id : str
            Entity ID (e.g., "light.living_room")

        Returns
        -------
        str
            Domain (e.g., "light")
        """
        return entity_id.split(".")[0] if "." in entity_id else ""

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Home Assistant API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.)
        endpoint : str
            API endpoint
        data : Optional[Dict[str, Any]]
            Request payload

        Returns
        -------
        Dict[str, Any]
            Response data
        """
        url = self._get_api_url(endpoint)
        ssl = None if self.config.verify_ssl else False

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                url,
                headers=self._headers,
                json=data,
                ssl=ssl
            ) as response:
                if response.status in (200, 201):
                    return await response.json()
                else:
                    error_text = await response.text()
                    logging.error(
                        f"Home Assistant API error: {response.status} - {error_text}"
                    )
                    return {"error": error_text, "status": response.status}

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Get current state of an entity.

        Parameters
        ----------
        entity_id : str
            Entity ID to query

        Returns
        -------
        Dict[str, Any]
            Entity state data
        """
        return await self._make_request("GET", f"states/{entity_id}")

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a Home Assistant service.

        Parameters
        ----------
        domain : str
            Service domain (e.g., "light", "switch", "climate")
        service : str
            Service name (e.g., "turn_on", "turn_off")
        service_data : Dict[str, Any]
            Service data including entity_id and parameters

        Returns
        -------
        Dict[str, Any]
            Service response
        """
        return await self._make_request(
            "POST",
            f"services/{domain}/{service}",
            service_data
        )

    async def connect(self, output_interface: HomeAssistantInput) -> HomeAssistantOutput:
        """
        Execute Home Assistant action based on input interface.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The input interface containing action details.

        Returns
        -------
        HomeAssistantOutput
            Result of the action execution
        """
        if not self.config.base_url or not self.config.access_token:
            logging.error("Home Assistant credentials not configured")
            return HomeAssistantOutput(
                success=False,
                state="",
                message="Home Assistant credentials not configured"
            )

        action = output_interface.action.lower()
        entity_id = output_interface.entity_id
        domain = self._get_domain_from_entity(entity_id)

        if not entity_id:
            return HomeAssistantOutput(
                success=False,
                state="",
                message="Entity ID is required"
            )

        try:
            logging.info(f"HomeAssistant action: {action} on {entity_id}")

            # Handle get_state action
            if action == "get_state":
                result = await self.get_state(entity_id)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state=result.get("state", "unknown"),
                        message=f"State of {entity_id}: {result.get('state', 'unknown')}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to get state: {result.get('error')}"
                    )

            # Handle turn_on action
            elif action == "turn_on":
                service_data = {"entity_id": entity_id}
                result = await self.call_service(domain, "turn_on", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state="on",
                        message=f"Successfully turned on {entity_id}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to turn on: {result.get('error')}"
                    )

            # Handle turn_off action
            elif action == "turn_off":
                service_data = {"entity_id": entity_id}
                result = await self.call_service(domain, "turn_off", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state="off",
                        message=f"Successfully turned off {entity_id}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to turn off: {result.get('error')}"
                    )

            # Handle set_brightness action for lights
            elif action == "set_brightness":
                if domain != "light":
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="Brightness can only be set for light entities"
                    )

                brightness = output_interface.brightness
                if brightness is None:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="Brightness value is required for set_brightness action"
                    )

                # Clamp brightness to valid range
                brightness = max(0, min(255, brightness))

                service_data = {
                    "entity_id": entity_id,
                    "brightness": brightness
                }
                result = await self.call_service("light", "turn_on", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state=f"brightness={brightness}",
                        message=f"Set brightness of {entity_id} to {brightness}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to set brightness: {result.get('error')}"
                    )

            # Handle set_color action for lights
            elif action == "set_color":
                if domain != "light":
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="Color can only be set for light entities"
                    )

                rgb_color = output_interface.rgb_color
                if rgb_color is None or len(rgb_color) != 3:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="RGB color (r, g, b) is required for set_color action"
                    )

                service_data = {
                    "entity_id": entity_id,
                    "rgb_color": list(rgb_color)
                }
                result = await self.call_service("light", "turn_on", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state=f"color={rgb_color}",
                        message=f"Set color of {entity_id} to RGB{rgb_color}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to set color: {result.get('error')}"
                    )

            # Handle set_temperature action for climate/thermostats
            elif action == "set_temperature":
                if domain != "climate":
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="Temperature can only be set for climate entities"
                    )

                temperature = output_interface.temperature
                if temperature is None:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message="Temperature value is required for set_temperature action"
                    )

                service_data = {
                    "entity_id": entity_id,
                    "temperature": temperature
                }
                result = await self.call_service("climate", "set_temperature", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state=f"temperature={temperature}",
                        message=f"Set temperature of {entity_id} to {temperature}°"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to set temperature: {result.get('error')}"
                    )

            # Handle toggle action
            elif action == "toggle":
                service_data = {"entity_id": entity_id}
                result = await self.call_service(domain, "toggle", service_data)
                if "error" not in result:
                    return HomeAssistantOutput(
                        success=True,
                        state="toggled",
                        message=f"Successfully toggled {entity_id}"
                    )
                else:
                    return HomeAssistantOutput(
                        success=False,
                        state="",
                        message=f"Failed to toggle: {result.get('error')}"
                    )

            else:
                return HomeAssistantOutput(
                    success=False,
                    state="",
                    message=f"Unknown action: {action}. Supported actions: turn_on, turn_off, toggle, set_brightness, set_color, set_temperature, get_state"
                )

        except Exception as e:
            logging.error(f"Failed to execute Home Assistant action: {str(e)}")
            return HomeAssistantOutput(
                success=False,
                state="",
                message=f"Error: {str(e)}"
            )
