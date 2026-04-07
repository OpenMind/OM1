from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class ArmAction(str, Enum):
    """
    Enumeration of possible arm actions.
    """

    IDLE = "idle"
    # Built-in actions (Unitree firmware, api_id=7106)
    LEFT_KISS = "left kiss"
    RIGHT_KISS = "right kiss"
    CLAP = "clap"
    HIGH_FIVE = "high five"
    HEART = "heart"
    HIGH_WAVE = "high wave"
    # Custom actions (g1_arm_action node, api_id=9001)
    SHAKE_HAND = "shake hand"
    FACE_WAVE = "face wave"
    HANDS_UP = "hands up"
    STAND_STILL = "stand still"
    SHOW_HAND = "show hand"
    WAVE = "wave"
    MOVE = "move"
    SHOW_HAND1 = "show hand1"
    SHOW_HAND2 = "show hand2"
    MY_GESTURE = "my gesture"


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
