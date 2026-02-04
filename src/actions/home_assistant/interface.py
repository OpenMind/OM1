from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class HomeAssistantControlInput:
    """Control a Home Assistant entity (smart device).

    This action is designed for simple, reliable device control via Home Assistant.

    Parameters
    ----------
    device : str
        A friendly device name/alias. Must exist in the action config mapping.
        Example: "living_room_light".
    command : str
        Command to execute. Supported: "on", "off", "toggle", "set".
    value : Optional[float]
        Optional numeric value for "set" (e.g. brightness %, temperature).
    """

    device: str
    command: str
    value: Optional[float] = None


@dataclass
class HomeAssistantControl(Interface[HomeAssistantControlInput, HomeAssistantControlInput]):
    """Controls smart devices via Home Assistant service calls."""

    input: HomeAssistantControlInput
    output: HomeAssistantControlInput
