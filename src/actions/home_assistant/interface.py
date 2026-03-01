from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class HADeviceType(str, Enum):
    """Supported Home Assistant device types."""

    LIGHT = "light"
    SWITCH = "switch"
    CLIMATE = "climate"


class HAAction(str, Enum):
    """Supported Home Assistant actions."""

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
    device_type : HADeviceType
        The type of device to control (light, switch, climate).
    entity_id : str
        The Home Assistant entity ID (e.g. light.living_room).
    action : HAAction
        The action to perform on the device.
    brightness : int
        Brightness level for lights (0-255).
    color : str
        Color name for lights (e.g. red, blue, green).
    temperature : float
        Target temperature for climate devices in Celsius.
    """

    device_type: HADeviceType = HADeviceType.LIGHT
    entity_id: str = ""
    action: HAAction = HAAction.TURN_ON
    brightness: int = 255
    color: str = ""
    temperature: float = 22.0


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantInput]):
    """
    This action allows the robot to control smart home devices via Home Assistant.

    Effect: Controls IoT devices including lights (on/off, brightness, color),
    switches (on/off), and thermostats (temperature setting) through the
    Home Assistant REST API.
    """

    input: HomeAssistantInput
    output: HomeAssistantInput


COLOR_MAP: dict[str, list[int]] = {
    "red": [0, 100],
    "green": [120, 100],
    "blue": [240, 100],
    "yellow": [60, 100],
    "orange": [30, 100],
    "purple": [270, 100],
    "pink": [300, 100],
    "white": [0, 0],
    "warm white": [30, 20],
    "cool white": [200, 10],
    "cyan": [180, 100],
}
