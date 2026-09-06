from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class FallRecoveryAction(str, Enum):
    """Supported fall recovery actions."""

    STAND_UP = "stand_up"
    EMERGENCY_STOP = "emergency_stop"
    ALERT_OPERATOR = "alert_operator"


@dataclass
class FallRecoveryInput:
    """
    Input interface for the FallRecovery action.

    Parameters
    ----------
    action : FallRecoveryAction
        The recovery action to perform.
    message : str
        Optional message describing the situation.
    """

    action: FallRecoveryAction = FallRecoveryAction.STAND_UP
    message: str = ""


@dataclass
class FallRecovery(Interface[FallRecoveryInput, FallRecoveryInput]):
    """
    This action allows the robot to recover from a fall or impact event.

    Effect: Executes fall recovery procedures including standing up,
    emergency stop, or alerting the operator. Triggered automatically
    by IMU fall detection or manually via LLM decision.
    """

    input: FallRecoveryInput
    output: FallRecoveryInput
