from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class RememberLocationInput:
    """
    Input payload for remembering/saving a named location.

    The 'action' field contains the location name to save (e.g. "kitchen", "office", "living room").
    The 'description' field is optional additional information about the location.

    Examples
    --------
    - User says: "Remember this location as kitchen" → action = "kitchen"
    - User says: "Save this spot as my office" → action = "office"
    - User says: "Remember this place as the charging station" → action = "charging station"
    """

    action: str
    description: Optional[str] = ""


@dataclass
class RememberLocation(Interface[RememberLocationInput, RememberLocationInput]):
    """
    Save/remember the robot's current location with a name.

    The 'action' field should contain the location name.
    Extract the location name from user commands like "remember this as [name]" or "save this location as [name]".
    """

    input: RememberLocationInput
    output: RememberLocationInput
