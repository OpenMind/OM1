from dataclasses import dataclass

from actions.base import Interface


@dataclass
class NavigateLocationInput:
    """
    Input payload for navigating to a stored location.

    CRITICAL: The 'action' field must contain ONLY the location name from the saved locations list.
    DO NOT include phrases like "go to", "navigate to", "move to", etc.

    Examples (assuming saved locations are: kitchen, sofa, table):
    - User says: "Go to the kitchen" → action = "kitchen"
    - User says: "Navigate to sofa" → action = "sofa"
    - User says: "Take me to the table" → action = "table"
    - User says: "Move to kitchen" → action = "kitchen"

    The action value must EXACTLY match one of the saved location names.
    """

    action: str


@dataclass
class NavigateLocation(Interface[NavigateLocationInput, NavigateLocationInput]):
    """
    Navigate to a saved location by name.

    Use ONLY the location name (like "kitchen", "sofa", "table") in the action field.
    Extract the location name from the user's command and use it directly.
    Do NOT include command phrases like "go to" or "navigate to".
    """

    input: NavigateLocationInput
    output: NavigateLocationInput
