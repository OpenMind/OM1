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

    Parameters
    ----------
    action : MovementAction
        The movement action to execute. Must be one of the predefined
        movement actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARDS, or STAND_STILL.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Turtle robot movement control action.

    This action provides control over turtle robot movement, supporting
    directional movement commands and stationary states. The action interface
    enables precise control of robot orientation and forward/backward motion.

    The movement actions include:
    - TURN_LEFT: Rotate the robot to the left
    - TURN_RIGHT: Rotate the robot to the right
    - MOVE_FORWARDS: Move the robot forward
    - STAND_STILL: Maintain current position without movement

    Important: Only safe movement values should be selected to prevent
    potential collisions or unsafe robot behavior.
    """

    input: MoveInput
    output: MoveInput
