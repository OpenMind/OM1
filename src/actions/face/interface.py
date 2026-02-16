from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class FaceAction(str, Enum):
    """
    Enumeration of possible facial expressions.
    """

    HAPPY = "happy"
    CONFUSED = "confused"
    CURIOUS = "curious"
    EXCITED = "excited"
    SAD = "sad"
    THINK = "think"


@dataclass
class FaceInput:
    """
    Input interface for the Face action.

    Parameters
    ----------
    action : FaceAction
        The facial expression to display. Must be one of the predefined expressions
        from the FaceAction enumeration (e.g., HAPPY, CONFUSED, CURIOUS, EXCITED,
        SAD, THINK).
    """

    action: FaceAction


@dataclass
class Face(Interface[FaceInput, FaceInput]):
    """
    Action interface for robot facial expression control.

    This action enables the robot to display various facial expressions to convey
    emotional states or cognitive states. The specific expression is determined
    by the FaceAction enum value provided in the input.

    Supported expressions include emotional states (HAPPY, SAD, EXCITED) and
    cognitive states (THINK, CURIOUS, CONFUSED), allowing the robot to provide
    visual feedback that enhances human-robot interaction.
    """

    input: FaceInput
    output: FaceInput
