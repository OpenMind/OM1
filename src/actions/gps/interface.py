from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class GPSAction(str, Enum):
    """
    Enumeration of possible GPS actions.
    """

    SHARE_LOCATION = "share location"
    IDLE = "idle"


@dataclass
class GPSInput:
    """
    Input interface for the GPS action.
    """

    action: GPSAction


@dataclass
class GPS(Interface[GPSInput, GPSInput]):
    """
    GPS location to be shared by the agent.

    Effect: Allows the agent to share its location.
    """

    input: GPSInput
    output: GPSInput
