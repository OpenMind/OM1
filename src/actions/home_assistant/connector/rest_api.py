"""
Home Assistant REST API Connector for OM1.

This module provides a connector to control Home Assistant devices using
the REST API. It supports lights, switches, thermostats, and other smart
home devices.
"""

import logging
import time
from typing import Any, Dict, Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import (
    HomeAssistantInput,
    HomeAssistantOutput,
)


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for Home Assistant REST API connector.

    Parameters
    ----------
    base_url : str
        Home Assistant URL (e.g., 'http://homeassistant.local:8123').
    access_token : str
        Long-lived access token from Home Assistant.
    verify_ssl : bool
        Whether to verify SSL certificates.
    timeout : int
        Request timeout in seconds.
    """

    base_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Home Assistant base URL",
    )
    access_token: str = Field(
        default="",
        description="Long-lived access token from Home Assistant",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )
    timeout: int = Field(
        default=10,
        description="Request timeout in seconds",
    )


class HomeAssistantRESTConnector(
    ActionConnector[HomeAssistantConfig, HomeAssistantInput]
):
    """
    A connector for controlling Home Assistant devices via REST API.

    This connector supports:
    - Lights: on/off, brightness, color (RGB)
    - Switches: on/off, toggle
    - Thermostats: temperature setting, HVAC mode
    - Covers: open/close
    - Fans: on/off, speed
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
        self.base_url = config.base_url.rstrip("/")
        self.access_token = config.access_token
        self.verify_ssl = config.verify_ssl
        self.timeout = config.timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_result: Optional[HomeAssistantOutput] = None

        if not self.access_token:
            logging.warning(
                "HomeAssistant REST connector: access_token is empty. "
                "API requests will fail. Set a long-lived access token "
                "from your Home Assistant instance."
            )

        logging.info(
            f"HomeAssistant REST connector initialized for {self.base_url}"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Home Assistant API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_headers(),
            )
        return self._session

    async def _call_service(
        self,
        domain: str,
        service: str,
        service_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call a Home Assistant service.

        Parameters
        ----------
        domain : str
            Service domain (e.g., 'light', 'switch', 'climate').
        service : str
            Service name (e.g., 'turn_on', 'turn_off').
        service_data : Dict[str, Any]
            Service data including entity_id and other parameters.

        Returns
        -------
        Dict[str, Any]
            Response from Home Assistant.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/services/{domain}/{service}"

        try:
            async with session.post(url, json=service_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "data": result}
                else:
                    error_text = await response.text()
                    logging.error(
                        f"Home Assistant API error: {response.status} - {error_text}"
                    )
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                    }
        except aiohttp.ClientError as e:
            logging.error(f"Home Assistant connection error: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error calling Home Assistant: {e}")
            return {"success": False, "error": str(e)}

    async def _get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an entity.

        Parameters
        ----------
        entity_id : str
            Home Assistant entity ID.

        Returns
        -------
        Optional[Dict[str, Any]]
            Entity state or None if not found.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/states/{entity_id}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.warning(f"Could not get state for {entity_id}")
                    return None
        except Exception as e:
            logging.error(f"Error getting entity state: {e}")
            return None

    async def _control_light(
        self,
        entity_id: str,
        action: str,
        brightness: Optional[int] = None,
        color_rgb: Optional[str] = None,
    ) -> HomeAssistantOutput:
        """Control a light device."""
        service_data: Dict[str, Any] = {"entity_id": entity_id}

        if action in ("turn_on", "on"):
            service = "turn_on"
            if brightness is not None:
                service_data["brightness"] = max(0, min(255, brightness))
            if color_rgb:
                try:
                    rgb = [int(x.strip()) for x in color_rgb.split(",")]
                    if len(rgb) == 3:
                        if all(0 <= v <= 255 for v in rgb):
                            service_data["rgb_color"] = rgb
                        else:
                            logging.warning(
                                f"RGB values out of range (0-255): {color_rgb}"
                            )
                except ValueError:
                    logging.warning(f"Invalid RGB color format: {color_rgb}")
        elif action in ("turn_off", "off"):
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        elif action == "brightness" and brightness is not None:
            service = "turn_on"
            service_data["brightness"] = max(0, min(255, brightness))
        elif action == "color" and color_rgb:
            service = "turn_on"
            try:
                rgb = [int(x.strip()) for x in color_rgb.split(",")]
                if len(rgb) == 3:
                    if all(0 <= v <= 255 for v in rgb):
                        service_data["rgb_color"] = rgb
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

        result = await self._call_service("light", service, service_data)

        if result["success"]:
            state = await self._get_entity_state(entity_id)
            new_state = state.get("state") if state else None
            return HomeAssistantOutput(
                success=True,
                message=f"Light {entity_id} {service} successful",
                entity_id=entity_id,
                new_state=new_state,
            )
        else:
            return HomeAssistantOutput(
                success=False,
                message=result.get("error", "Unknown error"),
                entity_id=entity_id,
            )

    async def _control_switch(
        self,
        entity_id: str,
        action: str,
    ) -> HomeAssistantOutput:
        """Control a switch device."""
        if action in ("turn_on", "on"):
            service = "turn_on"
        elif action in ("turn_off", "off"):
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown switch action: {action}",
                entity_id=entity_id,
            )

        result = await self._call_service(
            "switch", service, {"entity_id": entity_id}
        )

        if result["success"]:
            state = await self._get_entity_state(entity_id)
            new_state = state.get("state") if state else None
            return HomeAssistantOutput(
                success=True,
                message=f"Switch {entity_id} {service} successful",
                entity_id=entity_id,
                new_state=new_state,
            )
        else:
            return HomeAssistantOutput(
                success=False,
                message=result.get("error", "Unknown error"),
                entity_id=entity_id,
            )

    async def _control_thermostat(
        self,
        entity_id: str,
        action: str,
        temperature: Optional[float] = None,
        hvac_mode: Optional[str] = None,
    ) -> HomeAssistantOutput:
        """Control a climate/thermostat device."""
        service_data: Dict[str, Any] = {"entity_id": entity_id}

        if action == "set_temperature" and temperature is not None:
            service = "set_temperature"
            service_data["temperature"] = temperature
        elif action == "set_hvac_mode" and hvac_mode:
            service = "set_hvac_mode"
            service_data["hvac_mode"] = hvac_mode
        elif action in ("turn_off", "off"):
            service = "set_hvac_mode"
            service_data["hvac_mode"] = "off"
        elif action in ("turn_on", "on"):
            service = "set_hvac_mode"
            service_data["hvac_mode"] = hvac_mode or "auto"
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown thermostat action: {action}",
                entity_id=entity_id,
            )

        result = await self._call_service("climate", service, service_data)

        if result["success"]:
            state = await self._get_entity_state(entity_id)
            new_state = state.get("state") if state else None
            return HomeAssistantOutput(
                success=True,
                message=f"Thermostat {entity_id} {service} successful",
                entity_id=entity_id,
                new_state=new_state,
            )
        else:
            return HomeAssistantOutput(
                success=False,
                message=result.get("error", "Unknown error"),
                entity_id=entity_id,
            )

    async def _control_cover(
        self,
        entity_id: str,
        action: str,
    ) -> HomeAssistantOutput:
        """Control a cover device (blinds, garage doors, etc.)."""
        if action in ("open", "turn_on", "on"):
            service = "open_cover"
        elif action in ("close", "turn_off", "off"):
            service = "close_cover"
        elif action == "stop":
            service = "stop_cover"
        elif action == "toggle":
            service = "toggle"
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown cover action: {action}",
                entity_id=entity_id,
            )

        result = await self._call_service(
            "cover", service, {"entity_id": entity_id}
        )

        if result["success"]:
            state = await self._get_entity_state(entity_id)
            new_state = state.get("state") if state else None
            return HomeAssistantOutput(
                success=True,
                message=f"Cover {entity_id} {service} successful",
                entity_id=entity_id,
                new_state=new_state,
            )
        else:
            return HomeAssistantOutput(
                success=False,
                message=result.get("error", "Unknown error"),
                entity_id=entity_id,
            )

    async def _control_fan(
        self,
        entity_id: str,
        action: str,
    ) -> HomeAssistantOutput:
        """Control a fan device."""
        if action in ("turn_on", "on"):
            service = "turn_on"
        elif action in ("turn_off", "off"):
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        else:
            return HomeAssistantOutput(
                success=False,
                message=f"Unknown fan action: {action}",
                entity_id=entity_id,
            )

        result = await self._call_service(
            "fan", service, {"entity_id": entity_id}
        )

        if result["success"]:
            state = await self._get_entity_state(entity_id)
            new_state = state.get("state") if state else None
            return HomeAssistantOutput(
                success=True,
                message=f"Fan {entity_id} {service} successful",
                entity_id=entity_id,
                new_state=new_state,
            )
        else:
            return HomeAssistantOutput(
                success=False,
                message=result.get("error", "Unknown error"),
                entity_id=entity_id,
            )

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Connect and execute the Home Assistant action.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The input containing the device type, entity ID, and action.
        """
        device_type = output_interface.device_type.lower()
        entity_id = output_interface.entity_id
        action = output_interface.action.lower()

        logging.info(
            f"HomeAssistant action: {device_type}/{action} on {entity_id}"
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
        elif device_type in ("cover", "covers", "blind", "blinds"):
            result = await self._control_cover(
                entity_id=entity_id,
                action=action,
            )
        elif device_type in ("fan", "fans"):
            result = await self._control_fan(
                entity_id=entity_id,
                action=action,
            )
        else:
            result = HomeAssistantOutput(
                success=False,
                message=f"Unsupported device type: {device_type}",
                entity_id=entity_id,
            )

        self._last_result = result

        if result.success:
            logging.info(f"HomeAssistant: {result.message}")
        else:
            logging.error(f"HomeAssistant error: {result.message}")

    def tick(self) -> None:
        """
        Tick method for periodic updates.
        """
        time.sleep(60)

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
