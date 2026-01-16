from dataclasses import dataclass

from actions.base import Interface


@dataclass
class IDLEInput:
    """
    Input interface for the Game Controller action.

    Parameters
    ----------
    action : str
        The game controller command to be executed. This field is not actively used
        as the connector directly reads hardware input from the game controller.
    """

    action: str


@dataclass
class GameController(Interface[IDLEInput, IDLEInput]):
    """
    This action allows manual control of Unitree GO2 robots using game controllers.

    Effect: Enables teleoperation through physical game controllers (Xbox Wireless Controller,
    Sony DualSense, or Sony DualSense Edge). D-pad controls movement (forward/backward/left/right),
    triggers control rotation (clockwise/counter-clockwise), and face buttons control stance
    (stand up/sit down).

    Note: This connector has been deprecated. The OM1 Orchestrator now automatically
    handles game controller input without requiring this action to be explicitly called.
    The connector remains for backward compatibility with older configurations.
    """

    input: IDLEInput
    output: IDLEInput
