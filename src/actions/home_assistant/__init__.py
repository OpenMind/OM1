"""Home Assistant integration for OM1 robot runtime.

This module provides actions to control IoT devices through Home Assistant.
Supported devices: lights, switches, thermostats.

Example configuration in JSON5:

    home_assistant: {
        base_url: "http://homeassistant.local:8123",
        token: "YOUR_LONG_LIVED_ACCESS_TOKEN"
    }

How to get a long-lived access token:
1. Open Home Assistant UI
2. Go to your profile (click on your username in the sidebar)
3. Scroll down to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Save the token value (it won't be shown again)
"""

from actions.home_assistant.connector.home_assistant_api import (
    HomeAssistantConfig,
    HomeAssistantConnector,
)
from actions.home_assistant.interface import (
    HomeAssistant,
    HomeAssistantAction,
    HomeAssistantDeviceType,
    HomeAssistantInput,
)

__all__ = [
    "HomeAssistant",
    "HomeAssistantAction",
    "HomeAssistantConfig",
    "HomeAssistantConnector",
    "HomeAssistantDeviceType",
    "HomeAssistantInput",
]
