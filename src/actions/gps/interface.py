from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class GPSAction(str, Enum):
    """
    Enumeration of possible GPS actions.

    The GPS action supports sharing the agent's current location
    or maintaining an idle state when no location sharing is required.
    """

    SHARE_LOCATION = "share location"
    IDLE = "idle"


@dataclass
class GPSInput:
    """
    Input interface for the GPS action.

    Parameters
    ----------
    action : GPSAction
        The GPS action to perform. Can be either SHARE_LOCATION to
        share the agent's current location, or IDLE to maintain an
        inactive state.
    """

    action: GPSAction


@dataclass
class GPS(Interface[GPSInput, GPSInput]):
    """
    GPS location sharing action for the agent.

    This action enables the agent to share its current geographical
    location with users or other systems. The action supports two
    states: actively sharing location information or maintaining an
    idle state when location sharing is not required.

    The GPS action is essential for location-based services, navigation
    assistance, and spatial awareness in robotic applications.
    """

    input: GPSInput
    output: GPSInput
