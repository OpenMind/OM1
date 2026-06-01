from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class ReturnToBaseInput:
    """
    Input payload for the return to base action.

    The 'action' field should contain the return command (e.g. "return", "go home", "return to base").
    The 'base_location_name' is the saved location name of the base.
    It must match a location previously saved using remember_location.

    Examples
    --------
    - User says: "Return to base"        -> action = "return", base_location_name = "base"
    - User says: "Go home"               -> action = "return", base_location_name = "base"
    - User says: "Return to start"       -> action = "return", base_location_name = "start"
    - User says: "Go back to home base"  -> action = "return", base_location_name = "home_base"

    The 'base_location_name' must EXACTLY match one of the saved location names.
    If not provided, defaults to "base".
    """

    action: str
    base_location_name: Optional[str] = "base"


@dataclass
class ReturnToBase(Interface[ReturnToBaseInput, ReturnToBaseInput]):
    """
    Navigate the robot back to its base location.

    Use this action when the user wants the robot to return home, go back to
    its starting point, or return to a designated safe location. The base
    location must have been previously saved using remember_location.

    Set 'action' to "return".
    Set 'base_location_name' to the saved location name of the base
    (default: "base").
    """

    input: ReturnToBaseInput
    output: ReturnToBaseInput
