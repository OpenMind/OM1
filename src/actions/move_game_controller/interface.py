from dataclasses import dataclass

from actions.base import Interface


@dataclass
class IDLEInput:
    """
    Input interface for the Game Controller action.
    """

    action: str


@dataclass
class GameController(Interface[IDLEInput, IDLEInput]):
    """
    Game controller interface.
    """

    input: IDLEInput
    output: IDLEInput
