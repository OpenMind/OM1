from dataclasses import dataclass

from actions.base import Interface


@dataclass
class HomeAssistantInput:
    """Input interface for Home Assistant smart device control.

    OM1 action inputs typically accept a single `action` string. This action expects
    `action` to be JSON so the orchestrator can pass structured arguments.

    Expected JSON shape (examples):

    - Turn on a light:
      {"device": "living_room_light", "command": "on"}

    - Set brightness (percent):
      {"device": "living_room_light", "command": "set", "value": 50}

    Parameters
    ----------
    action : str
        JSON string containing {device, command, value?}.
    """

    action: str = ""


@dataclass
class HomeAssistant(Interface[HomeAssistantInput, HomeAssistantInput]):
    """Control smart devices via Home Assistant."""

    input: HomeAssistantInput
    output: HomeAssistantInput
