from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """Movement action types for web simulator."""

    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    MOVE_FORWARDS = "move forwards"
    MOVE_BACK = "move back"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"


@dataclass
class MoveInput:
    """Input interface for move web sim action."""

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """Move action interface for web simulator."""

    input: MoveInput
    output: MoveInput
