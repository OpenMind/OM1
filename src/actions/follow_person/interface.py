from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actions.base import Interface


class FollowMode(str, Enum):
    """
    Enumeration of follow modes.
    """

    BY_NAME = "by_name"  # Follow a specific person by name
    NEAREST = "nearest"  # Follow the nearest person
    LAST_SEEN = "last_seen"  # Follow the last seen person
    STOP = "stop"  # Stop following


@dataclass
class FollowPersonInput:
    """
    Input interface for the FollowPerson action.

    Parameters
    ----------
    action : str
        The person identifier to follow. Can be:
        - A person's name (e.g., "alice", "bob", "wendy")
        - "nearest" to follow the nearest person
        - "last_seen" to follow the last seen person
        - "stop" to stop following
    distance : float, optional
        Desired following distance in meters (default: 1.5).
        Range: 0.5 to 5.0 meters.
    speed : float, optional
        Following speed multiplier (0.0 to 1.0, default: 0.5).
        0.0 = no movement, 1.0 = maximum speed.
    stop_on_arrival : bool, optional
        Whether to stop when reaching the target distance (default: True).
    timeout_sec : float, optional
        Maximum time to follow before stopping if person is lost (default: 30.0).
    """

    action: str
    distance: float = 1.5
    speed: float = 0.5
    stop_on_arrival: bool = True
    timeout_sec: float = 30.0


@dataclass
class FollowPerson(Interface[FollowPersonInput, FollowPersonInput]):
    """
    This action allows the robot to follow a specific person.

    The robot will use visual recognition to identify the target person and
    maintain a safe following distance while moving. This is useful for
    home assistance, tour guiding, or companion robot scenarios.

    The robot continuously tracks the target person's position and adjusts
    its movement to maintain the desired distance. If the person is lost
    for more than the timeout period, following will stop automatically.

    Examples:
    - "Follow Alice" → action = "alice"
    - "Follow the nearest person" → action = "nearest"
    - "Follow me" → action = "last_seen"
    - "Stop following" → action = "stop"
    - "Follow Bob at 2 meters" → action = "bob", distance = 2.0
    """

    input: FollowPersonInput
    output: FollowPersonInput

