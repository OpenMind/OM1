from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SpeakInput:
    action: str


@dataclass
class Speak(Interface[SpeakInput, SpeakInput]):
    """
    This action allows you to speak.
    """

    input: SpeakInput
    output: SpeakInput
