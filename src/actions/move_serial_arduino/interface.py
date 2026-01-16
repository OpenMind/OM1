from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

    BE_STILL = "be still"
    JUMP_SMALL = "small jump"
    JUMP_MEDIUM = "medium jump"
    JUMP_BIG = "big jump"


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
