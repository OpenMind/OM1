from dataclasses import dataclass

from actions.base import Interface


@dataclass
class BittleInput:
    """
    Empty input for a configured Petoi Bittle command.
    """


@dataclass
class Bittle(Interface[BittleInput, BittleInput]):
    """
    Send one configured Petoi Bittle ASCII token over BLE.
    """

    input: BittleInput
    output: BittleInput
