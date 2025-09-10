from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    STAND_STILL = "stand still"
    SIT = "sit"
    DANCE = "dance"
    SHAKE_PAW = "shake paw"
    WALK = "walk"
    WALK_BACK = "walk back"
    RUN = "run"
    JUMP = "jump"
    WAG_TAIL = "wag tail"


@dataclass
class MoveInput:
    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    This action allows you to move.
    """

    input: MoveInput
    output: MoveInput
