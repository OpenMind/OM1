"""
Home Assistant Action Interface for OM1.

This module defines the interface for controlling Home Assistant devices
including lights, switches, thermostats, and other smart home devices.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actions.base import Interface


class DeviceType(Enum):
    """Supported Home Assistant device types."""

    LIGHT = "light"
    SWITCH = "switch"
    THERMOSTAT = "climate"
    COVER = "cover"
    FAN = "fan"


class LightAction(Enum):
    """Available actions for light devices."""

    ON = "turn_on"
    OFF = "turn_off"
    TOGGLE = "toggle"
    BRIGHTNESS = "brightness"
    COLOR = "color"


class SwitchAction(Enum):
    """Available actions for switch devices."""

    ON = "turn_on"
    OFF = "turn_off"
    TOGGLE = "toggle"


class ThermostatAction(Enum):
    """Available actions for thermostat/climate devices."""

    SET_TEMPERATURE = "set_temperature"
    SET_HVAC_MODE = "set_hvac_mode"
    SET_FAN_MODE = "set_fan_mode"


class HVACMode(Enum):
    """HVAC modes for thermostats."""

    OFF = "off"
    HEAT = "heat"
    COOL = "cool"
    AUTO = "auto"
    DRY = "dry"
    FAN_ONLY = "fan_only"


@dataclass
class HomeAssistantInput:
    """
    Input interface for the Home Assistant action.

    Parameters
    ----------
    device_type : str
        Type of device: 'light', 'switch', 'climate' (thermostat), 'cover', 'fan'.
    entity_id : str
        Home Assistant entity ID (e.g., 'light.living_room', 'switch.bedroom').
    action : str
        Action to perform: 'turn_on', 'turn_off', 'toggle', 'brightness', 'color',
        'set_temperature', 'set_hvac_mode'.
    brightness : Optional[int]
        Brightness level (0-255) for lights.
    color_rgb : Optional[str]
        RGB color as comma-separated values (e.g., '255,0,0' for red).
    temperature : Optional[float]
        Target temperature for thermostats.
    hvac_mode : Optional[str]
        HVAC mode: 'off', 'heat', 'cool', 'auto', 'dry', 'fan_only'.
    """

    device_type: str
    entity_id: str
    action: str
    brightness: Optional[int] = None
    color_rgb: Optional[str] = None
    temperature: Optional[float] = None
    hvac_mode: Optional[str] = None


@dataclass
class HomeAssistantOutput:
    """
    Output interface for the Home Assistant action.

    Parameters
    ----------
    success : bool
        Whether the action was successful.
    message : str
        Response message from Home Assistant.
    entity_id : str
        The entity ID that was controlled.
    new_state : Optional[str]
        The new state of the device after the action.
    """

    success: bool
    message: str
    entity_id: str
    new_state: Optional[str] = None


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantOutput]):
    """
    This action controls Home Assistant smart devices including lights, switches,
    thermostats, and other IoT devices. Use this to turn devices on/off, adjust
    brightness/color for lights, or set temperature for thermostats.
    """

    input: HomeAssistantInput
    output: HomeAssistantOutput
