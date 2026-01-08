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
    STAND_STILL = "stand still"


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
