from dataclasses import dataclass

from actions.base import Interface


@dataclass
class HomeAssistantInput:
    """Input for Home Assistant actions. The action field contains the command string."""

    action: str


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantInput]):
    """
    Control smart home devices via Home Assistant.
    Commands: turn on/off switches and lights, set thermostat temperature.
    """

    input: HomeAssistantInput
    output: HomeAssistantInput
