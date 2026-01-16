from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

    STAND_UP = "stand up"
    SIT = "sit"
    SHAKE_PAW = "shake paw"
    DANCE = "dance"
    STRETCH = "stretch"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"


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
