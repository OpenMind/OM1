from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class DockChargingInput:
    """
    Input payload for the dock charging action.

    The 'action' field should contain the dock command (e.g. "dock", "charge", "go charge").
    The 'dock_location_name' is the saved location name of the charging dock.
    It must match a location previously saved using remember_location.

    Examples
    --------
    - User says: "Go dock"                    -> action = "dock", dock_location_name = "charging_dock"
    - User says: "Go charge yourself"         -> action = "dock", dock_location_name = "charging_dock"
    - User says: "Return to charging station" -> action = "dock", dock_location_name = "charging_dock"
    - User says: "Go to dock2 to charge"      -> action = "dock", dock_location_name = "dock2"

    The 'dock_location_name' must EXACTLY match one of the saved location names.
    If not provided, defaults to "charging_dock".
    """

    action: str
    dock_location_name: Optional[str] = "charging_dock"


@dataclass
class DockCharging(Interface[DockChargingInput, DockChargingInput]):
    """
    Navigate the robot to its charging dock and initiate charging.

    Use this action when the user wants the robot to go charge, dock, or return
    to its charging station. The dock location must have been previously saved
    using remember_location.

    Set 'action' to "dock".
    Set 'dock_location_name' to the saved location name of the charging dock
    (default: "charging_dock").
    """

    input: DockChargingInput
    output: DockChargingInput
