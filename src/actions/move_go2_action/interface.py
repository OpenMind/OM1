from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class Action(str, Enum):
    SHAKE_PAW = "shake paw"
    DANCE = "dance"
    STRETCH = "stretch"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"


@dataclass
class ActionInput:
    action: Action


@dataclass
class Move(Interface[ActionInput, ActionInput]):
    """
    This action allows you to move. Important: pick only safe values.
    """

    input: ActionInput
    output: ActionInput
