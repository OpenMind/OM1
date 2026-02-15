from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

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
    """
    Input interface for the Move action.

    Parameters
    ----------
    action : MovementAction
        The movement action to execute. Must be one of the predefined movement
        actions from the MovementAction enumeration (e.g., STAND_STILL, SIT,
        DANCE, WALK, RUN, JUMP).
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Action interface for robot movement commands.

    This action enables the robot to perform various predefined movement behaviors
    such as standing still, sitting, dancing, walking, running, and jumping. The
    specific movement is determined by the MovementAction enum value provided
    in the input.

    The action supports both static poses (e.g., STAND_STILL, SIT) and dynamic
    movements (e.g., WALK, RUN, JUMP), allowing for flexible robot behavior
    control through the LLM interface.
    """

    input: MoveInput
    output: MoveInput
