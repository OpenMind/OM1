from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SpeakInput:
    """
    Input interface for the Speak action.

    Parameters
    ----------
    action : str
        The text to be spoken.
    """

    action: str


@dataclass
class Speak(Interface[SpeakInput, SpeakInput]):
    """
    This action makes the robot speak a given text.
    """

    input: SpeakInput
    output: SpeakInput
