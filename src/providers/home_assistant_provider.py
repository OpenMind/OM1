import requests
from typing import Dict
from .io_provider import IOProvider


class HomeAssistantProvider:
    """
    REST-based Home Assistant provider for controlling IoT devices
    via OM1.

    Supports:
    - On / Off (switches, lights)
    - Brightness (lights)
    - Temperature (thermostats)
    """

    def __init__(self, base_url: str, access_token: str):
        """
        Parameters
        ----------
        base_url : str
            Base URL of the Home Assistant instance (e.g. http://localhost:8123)
        access_token : str
            Long-lived access token from Home Assistant
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        self.io = IOProvider()

    def _call_service(self, domain: str, service: str, data: Dict):
        """
        Call a Home Assistant service via REST API.
        """
        url = f"{self.base_url}/api/services/{domain}/{service}"
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    # ----------------------
    # BASIC CONTROLS
    # ----------------------

    def turn_on(self, entity_id: str):
        """
        Turn on a switch or light.
        """
        domain = entity_id.split(".")[0]
        self._call_service(domain, "turn_on", {"entity_id": entity_id})
        self.io.add_input(
            "device_action",
            f"{entity_id} turned ON",
            None,
        )

    def turn_off(self, entity_id: str):
        """
        Turn off a switch or light.
        """
        domain = entity_id.split(".")[0]
        self._call_service(domain, "turn_off", {"entity_id": entity_id})
        self.io.add_input(
            "device_action",
            f"{entity_id} turned OFF",
            None,
        )

    def set_brightness(self, entity_id: str, brightness: int):
        """
        Set brightness for a light (0–255).
        """
        self._call_service(
            "light",
            "turn_on",
            {
                "entity_id": entity_id,
                "brightness": brightness,
            },
        )
        self.io.add_input(
            "device_action",
            f"{entity_id} brightness set to {brightness}",
            None,
        )

    def set_temperature(self, entity_id: str, temperature: float):
        """
        Set temperature for a thermostat.
        """
        self._call_service(
            "climate",
            "set_temperature",
            {
                "entity_id": entity_id,
                "temperature": temperature,
            },
        )
        self.io.add_input(
            "device_action",
            f"{entity_id} temperature set to {temperature}",
            None,
        )
