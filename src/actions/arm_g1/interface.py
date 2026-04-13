from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class ArmAction(str, Enum):
    """
    Enumeration of possible arm actions.

    Built-in actions are handled by Unitree firmware (api_id=7106).
    Custom actions are handled by the g1_arm_action ROS2 node (api_id=9001).
    """

    IDLE = "idle"
    # LEFT_KISS = "left kiss"
    # RIGHT_KISS = "right kiss"
    # CLAP = "clap"
    # HIGH_FIVE = "high five"
    # HEART = "heart"
    # HIGH_WAVE = "high wave"
    SHAKE_HAND = "shake_hand"
    FACE_WAVE = "face_wave"
    HANDS_UP = "hands_up"
    STAND_STILL = "stand_still"
    SHOW_HAND = "show_hand"
    DO_PAYMENT = "do_payment"
    DOWN_PAYMENT = "down_payment"
    WAVE = "wave"
    # MOVE = "move"


@dataclass
class ArmInput:
    """
    Input interface for the Arm action.

    Parameters
    ----------
    action : ArmAction
        The arm movement to perform. Must be one of the predefined actions
        from the ArmAction enumeration.
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
