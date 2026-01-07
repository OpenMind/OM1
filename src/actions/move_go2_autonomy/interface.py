from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    MOVE_FORWARDS = "move forwards"
    MOVE_BACK = "move back"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"
    # STAND_UP = "stand up"
    # SIT = "sit"
    # SHAKE_PAW = "shake paw"
    # DANCE = "dance"
    # STRETCH = "stretch"
    # STAND_STILL = "stand still"
    # DO_NOTHING = "stand still"


@dataclass
class MoveInput:
    """
    Input interface for the Move action.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    This action allows you to move. Important: pick only safe values.
    """

    input: MoveInput
    output: MoveInput
