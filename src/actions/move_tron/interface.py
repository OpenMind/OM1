from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions for Tron robots.

    This enumeration defines the supported movement commands that can be
    executed by Tron robots through the WebSocket-based SDK interface.
    The actions include directional movements (forward, backward, turning)
    and static states (standing still).

    Attributes
    ----------
    TURN_LEFT : str
        Command to turn the robot left.
    TURN_RIGHT : str
        Command to turn the robot right.
    MOVE_FORWARDS : str
        Command to move the robot forward.
    MOVE_BACK : str
        Command to move the robot backward.
    STAND_STILL : str
        Command to keep the robot in a stationary position.
    DO_NOTHING : str
        Alias for STAND_STILL, maintains the robot's current position.
    """

    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    MOVE_FORWARDS = "move forwards"
    MOVE_BACK = "move back"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"


@dataclass
class MoveInput:
    """
    Input interface for the Move action on Tron robots.

    This dataclass represents the input parameters required to execute
    a movement command on a Tron robot. The action parameter specifies
    the type of movement to be performed.

    Parameters
    ----------
    action : MovementAction
        The movement action to be performed by the Tron robot. Must be
        one of the predefined movement actions in the MovementAction
        enumeration. The action is sent to the robot via WebSocket
        connection and converted to velocity commands (x, y, z).

    Notes
    -----
    The action value is processed by the MoveTronSDKConnector, which
    converts the string action into velocity commands:
    - "move forwards" -> x = 0.5
    - "move back" -> x = -0.5
    - "turn left" -> z = 0.5
    - "turn right" -> z = -0.5
    - "stand still" -> x = 0.0, y = 0.0, z = 0.0
    """

    action: MovementAction

    def __post_init__(self) -> None:
        """
        Validate the action parameter after initialization.

        Raises
        ------
        ValueError
            If the action is None or not a valid MovementAction enum value.
        """
        if self.action is None:
            raise ValueError("action parameter cannot be None")
        if not isinstance(self.action, MovementAction):
            raise ValueError(
                f"action must be a MovementAction enum value, got {type(self.action)}"
            )


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Movement action interface for Tron robots.

    This action allows the agent to control Tron robot movements through
    a WebSocket-based SDK interface. The action supports directional
    movements (forward, backward, turning) and static states (standing still).

    The action uses the MoveTronSDKConnector to communicate with the robot's
    control system, converting high-level movement commands into velocity
    commands that are transmitted via WebSocket.

    Parameters
    ----------
    input : MoveInput
        The input containing the movement action to execute on the Tron robot.
    output : MoveInput
        The output mirroring the input action (passthrough interface).

    Notes
    -----
    The action requires a valid WebSocket connection to the Tron SDK server
    (default: ws://10.192.1.2:5000) and a robot serial number (accid) for
    proper operation. The connector handles the conversion of movement
    actions to velocity commands and manages the WebSocket communication.
    """

    input: MoveInput
    output: MoveInput
