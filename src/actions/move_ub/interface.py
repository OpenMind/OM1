from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

    WAVE = "wave"
    BOW = "bow"
    CROUCH = "crouch"
    COME = "come on"
    STAND_STILL = "reset"
    DO_NOTHING = "reset"
    WALK_FORWARD = "walk forward"
    WALK_BACKWARD = "walk backward"
    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    LOOK_LEFT = "look left"
    LOOK_RIGHT = "look right"
    WALK_LEFT = "walk left"
    WALK_RIGHT = "walk right"
    WAKAWAKA = "WakaWaka"
    HUG = "Hug"
    RAISE_RIGHT_HAND = "RaiseRightHand"
    PUSH_UP = "PushUp"


@dataclass
class MoveInput:
    """
    Input interface for the Move action.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    A movement to be performed by the agent.

    Effect: Allows the agent to move.
    """

    input: MoveInput
    output: MoveInput
