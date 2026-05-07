from dataclasses import dataclass

from actions.base import Interface


@dataclass
class BittleCalibrationAdjustmentInput:
    """
    Input interface for adjusting one Petoi Bittle calibration servo.

    Parameters
    ----------
    servo_index : int
        Servo index to adjust. Valid Petoi UI-exposed servos are 0 and 8 through 15.
    degrees : int
        Integer adjustment in degrees. Valid range is -9 through 9.
    """

    servo_index: int
    degrees: int


@dataclass
class BittleCalibrationAdjustment(
    Interface[BittleCalibrationAdjustmentInput, BittleCalibrationAdjustmentInput]
):
    """
    Adjust one Petoi Bittle calibration servo over BLE.
    """

    input: BittleCalibrationAdjustmentInput
    output: BittleCalibrationAdjustmentInput
