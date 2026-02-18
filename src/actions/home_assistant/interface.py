from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actions.base import Interface


class HomeAssistantDeviceType(str, Enum):
    """
    Enumeration of supported Home Assistant device types.
    """

    LIGHT = "light"
    SWITCH = "switch"
    THERMOSTAT = "climate"


class HomeAssistantAction(str, Enum):
    """
    Enumeration of possible Home Assistant actions.
    """

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    SET_BRIGHTNESS = "set_brightness"
    SET_COLOR = "set_color"
    SET_TEMPERATURE = "set_temperature"


@dataclass
class HomeAssistantInput:
    """
    Input interface for the Home Assistant action.

    Parameters
    ----------
    device_type : HomeAssistantDeviceType
        Type of device to control (light, switch, climate)
    device_id : str
        Entity ID of the device in Home Assistant (e.g., "light.living_room")
    action : HomeAssistantAction
        Action to perform on the device
    brightness : Optional[int]
        Brightness level (0-255) for lights
    color : Optional[str]
        Color name or hex code for lights
    temperature : Optional[float]
        Temperature setting for thermostats
    """

    device_type: HomeAssistantDeviceType
    device_id: str
    action: HomeAssistantAction
    brightness: Optional[int] = None
    color: Optional[str] = None
    temperature: Optional[float] = None


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantInput]):
    """
    This action allows you to control Home Assistant IoT devices.
    
    Supported devices: lights, switches, thermostats.
    Supported actions: turn on/off, set brightness, set color, set temperature.
    """

    input: HomeAssistantInput
    output: HomeAssistantInput
