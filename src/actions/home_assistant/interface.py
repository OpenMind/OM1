from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class HomeAssistantInput:
    """
    Input interface for the HomeAssistant action.

    Parameters
    ----------
    device : str
        Device alias from the config (e.g. "living_room_light").
        Use the exact alias defined in the agent config, not the raw entity ID.
    command : str
        Command to execute. Universal commands: "on", "off", "toggle".
        Light: "set_brightness", "set_color", "set_color_temp".
        Climate: "set_temperature", "set_hvac_mode", "set_fan_mode", "set_preset".
        Lock: "lock", "unlock".
        Cover: "open", "close", "stop", "set_position".
        Media: "play", "pause", "media_stop", "volume_set", "volume_mute",
               "volume_unmute", "select_source".
        Fan: "set_percentage", "oscillate", "stop_oscillate".
        Vacuum: "start", "stop", "vacuum_pause", "return_to_base".
        Scene: "activate".
        Alarm: "arm_home", "arm_away", "arm_night", "disarm".
    value : Optional[float]
        Numeric parameter for commands that require one.
        Brightness (0-100), temperature, volume (0-100), position (0-100),
        fan percentage (0-100), color temperature in Kelvin.
    mode : Optional[str]
        String parameter for commands that require one.
        HVAC mode ("heat", "cool", "auto", "off"), fan mode, preset mode,
        media source name, color as hex string ("#FF0000").
    """

    device: str
    command: str
    value: Optional[float] = None
    mode: Optional[str] = None


@dataclass
class HomeAssistantControl(Interface[HomeAssistantInput, HomeAssistantInput]):
    """
    A Home Assistant action to control smart home devices.
    Effect: Allows the agent to control smart home devices including lights,
    switches, thermostats, locks, covers/blinds, media players, fans, vacuums,
    scenes, and alarms.
    Use the device alias (e.g. "living_room_light") and a command.
    Universal: "on", "off", "toggle".
    Light: "set_brightness" (value=0-100), "set_color" (mode="#FF0000"),
           "set_color_temp" (value=kelvin).
    Climate: "set_temperature" (value=degrees), "set_hvac_mode" (mode="heat"|"cool"|"auto"|"off"),
             "set_fan_mode" (mode), "set_preset" (mode).
    Lock: "lock", "unlock".
    Cover: "open", "close", "stop", "set_position" (value=0-100).
    Media: "play", "pause", "media_stop", "volume_set" (value=0-100),
           "volume_mute", "volume_unmute", "select_source" (mode=source_name).
    Fan: "set_percentage" (value=0-100), "oscillate", "stop_oscillate".
    Vacuum: "start", "stop", "vacuum_pause", "return_to_base".
    Scene: "activate". Alarm: "arm_home", "arm_away", "arm_night", "disarm".
    """

    input: HomeAssistantInput
    output: HomeAssistantInput
