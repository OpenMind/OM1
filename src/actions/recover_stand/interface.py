from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class RecoverAction(str, Enum):
    """
    Enumeration of possible recovery actions.
    """

    RECOVER = "recover"


@dataclass
class RecoverInput:
    """
    Input interface for the RecoverStand action.

    Parameters
    ----------
    action : RecoverAction
        The recovery action to execute. Must be one of the predefined actions
        from the RecoverAction enumeration (e.g., RECOVER).
    """

    action: RecoverAction


@dataclass
class RecoverStand(Interface[RecoverInput, RecoverInput]):
    """
    Use this action to stand back up after falling down.
    Call this when you detect that you have fallen over or are on the ground.
    """

    input: RecoverInput
    output: RecoverInput
