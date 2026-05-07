from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class BittleCalibrationOperation(str, Enum):
    """
    Petoi Bittle calibration operations.
    """

    ENTER = "enter calibration"
    ADJUST = "adjust servo"
    SAVE = "save calibration"


@dataclass
class BittleCalibrationInput:
    """
    Input interface for Petoi Bittle calibration.

    Parameters
    ----------
    operation : BittleCalibrationOperation
        Calibration operation to run: enter calibration, adjust servo, or save calibration.
    servo_index : int
        Servo index for adjust servo. Valid Petoi UI-exposed servos are 0 and 8 through 15.
    degrees : int
        Integer adjustment in degrees for adjust servo. Valid range is -9 through 9.
    """

    operation: BittleCalibrationOperation
    servo_index: int = 0
    degrees: int = 0


@dataclass
class BittleCalibration(Interface[BittleCalibrationInput, BittleCalibrationInput]):
    """
    Run documented Petoi Bittle calibration commands over BLE: enter calibration, adjust servo, or save calibration.
    Servo adjustment is limited to servo indexes 0 and 8 through 15, with integer degree changes from -9 through 9.
    """

    input: BittleCalibrationInput
    output: BittleCalibrationInput
