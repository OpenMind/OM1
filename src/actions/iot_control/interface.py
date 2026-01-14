from dataclasses import dataclass
from enum import Enum
from actions.base import Interface


class IoTAction(str, Enum):
    """Available IoT control actions."""
    LIGHTS_ON = "lights on"
    LIGHTS_OFF = "lights off"
    LIGHTS_TOGGLE = "toggle lights"
    FAN_ON = "fan on"
    FAN_OFF = "fan off"


@dataclass
class IoTInput:
    """
    Input interface for IoT control action.
    
    Supports basic home automation commands like lights and fan control.
    Works with Home Assistant demo entities (no physical devices required).
    """
    action: str = "lights on"
    device: str = "all"  # "all", "living_room", "bedroom", etc.
    
    def __repr__(self):
        return f"\n🏠 [IoT] EXECUTING: {self.action} ({self.device})\n"


class IoTControl(Interface):
    """IoT Control action interface."""
    input: IoTInput
    output: IoTInput
