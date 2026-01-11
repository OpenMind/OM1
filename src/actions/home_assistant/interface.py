from dataclasses import dataclass

from actions.base import Interface


@dataclass
class HomeAssistantInput:
    """Input for Home Assistant actions."""

    action: str


@dataclass
class HomeAssistantOutput:
    """Output from Home Assistant actions."""

    status: str
    message: str = ""


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantOutput]):
    """
    Control smart home devices and place orders via Home Assistant.
    Supports: switches, lights, thermostats, and order placement with crypto payment.
    Commands: 'turn on switch', 'set temperature 24', 'place order coffee 5 usdc'.
    """

    input: HomeAssistantInput
    output: HomeAssistantOutput
