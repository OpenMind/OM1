from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class ArmAction(str, Enum):
    """
    Enumeration of possible arm actions.
    """

    IDLE = "idle"
    LEFT_KISS = "left kiss"
    RIGHT_KISS = "right kiss"
    CLAP = "clap"
    HIGH_FIVE = "high five"
    SHAKE_HAND = "shake hand"
    HEART = "heart"
    HIGH_WAVE = "high wave"


@dataclass
class ArmInput:
    """
    Input interface for the Arm action.
    """

    action: ArmAction


@dataclass
class Arm(Interface[ArmInput, ArmInput]):
    """
    An arm movement to be performed by the agent.
    Effect: Allows the agent to perform arm movements.
    """

    input: ArmInput
    output: ArmInput
