from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class HomeAssistantInput:
    """
    Input interface for the Home Assistant action.

    Parameters
    ----------
    action : str
        The action to perform. Supported actions:
        - "turn_on" / "turn_off" for switches and lights
        - "set_brightness" for lights (requires brightness parameter)
        - "set_color" for lights (requires rgb_color parameter)
        - "set_temperature" for thermostats (requires temperature parameter)
        - "get_state" to query device state
    entity_id : str
        The Home Assistant entity ID (e.g., "light.living_room", "switch.fan", "climate.thermostat")
    brightness : Optional[int]
        Brightness level 0-255 for lights (used with set_brightness action)
    rgb_color : Optional[tuple]
        RGB color tuple (r, g, b) for lights (used with set_color action)
    temperature : Optional[float]
        Target temperature for thermostats (used with set_temperature action)
    """

    action: str = ""
    entity_id: str = ""
    brightness: Optional[int] = None
    rgb_color: Optional[tuple] = None
    temperature: Optional[float] = None


@dataclass
class HomeAssistantOutput:
    """
    Output interface for the Home Assistant action.

    Parameters
    ----------
    success : bool
        Whether the action was successful
    state : str
        Current state of the device after action
    message : str
        Human-readable message about the action result
    """

    success: bool = False
    state: str = ""
    message: str = ""


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantOutput]):
    """
    This action allows the robot to control smart home devices via Home Assistant.

    Effect: Sends commands to Home Assistant to control IoT devices like lights,
    switches, and thermostats. Supports turning devices on/off, adjusting brightness,
    changing colors, and setting temperatures.

    Supported device types:
    - Lights: on/off, brightness (0-255), RGB color
    - Switches: on/off
    - Climate/Thermostats: set target temperature

    Example usage:
    - Turn on living room light: action="turn_on", entity_id="light.living_room"
    - Set brightness to 50%: action="set_brightness", entity_id="light.bedroom", brightness=128
    - Set thermostat to 22°C: action="set_temperature", entity_id="climate.thermostat", temperature=22.0
    """

    input: HomeAssistantInput
    output: HomeAssistantOutput
