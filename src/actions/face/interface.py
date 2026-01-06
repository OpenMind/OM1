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
    """

    action: FaceAction


@dataclass
class Face(Interface[FaceInput, FaceInput]):
    """
    This action allows you to show facial expressions.
    """

    input: FaceInput
    output: FaceInput
