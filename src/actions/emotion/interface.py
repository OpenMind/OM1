from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class EmotionAction(str, Enum):
    """
    Enumeration of possible emotions.
    """

    HAPPY = "happy"
    SAD = "sad"
    MAD = "mad"
    CURIOUS = "curious"


@dataclass
class EmotionInput:
    """
    Input interface for the Emotion action.

    Parameters
    ----------
    action : EmotionAction
        The emotion to express. Must be one of the predefined emotions from the
        EmotionAction enumeration (e.g., HAPPY, SAD, MAD, CURIOUS).
    """

    action: EmotionAction


@dataclass
class Emotion(Interface[EmotionInput, EmotionInput]):
    """
    Action interface for robot emotion expression.

    This action enables the robot to express various emotional states through
    its behavior and appearance. The specific emotion is determined by the
    EmotionAction enum value provided in the input.

    The emotion system allows the robot to communicate its internal state or
    respond empathetically to user interactions, enhancing the naturalness
    of human-robot communication.
    """

    input: EmotionInput
    output: EmotionInput
