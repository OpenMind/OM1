from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class EmotionAction(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    MAD = "mad"
    CURIOUS = "curious"


@dataclass
class EmotionInput:
    action: EmotionAction


@dataclass
class Emotion(Interface[EmotionInput, EmotionInput]):
    """
    This action allows you to show your emotions.
    """

    input: EmotionInput
    output: EmotionInput
