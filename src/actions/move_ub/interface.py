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

    Parameters
    ----------
    action : MovementAction
        The movement action to be performed by the agent. Must be one of the
        predefined movement actions in the MovementAction enumeration.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Movement action interface for Ubtech robots.

    This action allows the agent to perform various movement actions including
    gestures (wave, bow, hug), directional movements (walk forward/backward,
    turn left/right), and special actions (push-up, raise hand).

    The action supports both static poses (STAND_STILL, CROUCH) and dynamic
    movements (WALK_FORWARD, TURN_LEFT) for comprehensive robot control.

    Parameters
    ----------
    input : MoveInput
        The input containing the movement action to execute.
    output : MoveInput
        The output mirroring the input action (passthrough interface).
    """

    input: MoveInput
    output: MoveInput
