from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from actions.base import Interface

if TYPE_CHECKING:
    pass


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    """

    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    MOVE_FORWARDS = "move forwards"
    MOVE_BACK = "move back"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"
    # STAND_UP = "stand up"
    # SIT = "sit"
    # SHAKE_PAW = "shake paw"
    # DANCE = "dance"
    # STRETCH = "stretch"
    # STAND_STILL = "stand still"
    # DO_NOTHING = "stand still"


@dataclass
class MoveInput:
    """
    Input interface for the Move action.

    Parameters
    ----------
    action : MovementAction
        The movement action to be performed. Must be one of the
        predefined movement actions in the MovementAction enumeration.

    Raises
    ------
    ValueError
        If the action is not a valid MovementAction value.
    """

    action: MovementAction

    def __post_init__(self) -> None:
        """
        Validate that the action is a valid MovementAction.

        Raises
        ------
        ValueError
            If the action is not a valid MovementAction value.
        """
        if not isinstance(self.action, MovementAction):
            if isinstance(self.action, str):
                try:
                    # Try to convert string to enum
                    self.action = MovementAction(self.action)
                except ValueError:
                    valid_values = [e.value for e in MovementAction]
                    raise ValueError(
                        f"Invalid action '{self.action}'. "
                        f"Must be one of: {', '.join(valid_values)}"
                    )
            else:
                raise TypeError(
                    f"Action must be a MovementAction or str, got {type(self.action)}"
                )


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    This action allows you to move. Important: pick only safe values.

    Parameters
    ----------
    input : MoveInput
        The input containing the movement action to execute.
    output : MoveInput
        The output mirroring the input action (passthrough interface).
    """

    input: MoveInput = field(default_factory=lambda: MoveInput(action=MovementAction.STAND_STILL))
    output: MoveInput = field(default_factory=lambda: MoveInput(action=MovementAction.STAND_STILL))
