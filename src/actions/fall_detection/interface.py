from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class FallSeverity(str, Enum):
    """
    Enumeration of fall severity levels.
    """

    LOW = "low"  # Possible fall, needs attention
    MEDIUM = "medium"  # Likely fall, requires immediate check
    HIGH = "high"  # Confirmed fall, emergency situation


@dataclass
class FallDetectionInput:
    """
    Input interface for the FallDetection action.

    Parameters
    ----------
    severity : FallSeverity
        The severity level of the detected fall.
    location : str, optional
        Location where the fall was detected (e.g., "living room", "bedroom").
    person_name : str, optional
        Name of the person who fell (if known from face recognition).
    confidence : float, optional
        Confidence score of the fall detection (0.0 to 1.0).
    additional_info : str, optional
        Additional information about the fall detection.
    """

    severity: FallSeverity
    location: str = ""
    person_name: str = ""
    confidence: float = 0.0
    additional_info: str = ""


@dataclass
class FallDetection(Interface[FallDetectionInput, FallDetectionInput]):
    """
    This action allows the robot to detect and respond to human falls.

    The robot can detect when a person falls using computer vision and pose estimation.
    When a fall is detected, the robot can:
    - Send immediate alerts to caregivers or emergency services
    - Approach the person to check their condition
    - Provide verbal assistance and instructions
    - Record the incident for medical records

    This is particularly useful for:
    - Elderly care and monitoring
    - Post-surgery patient monitoring
    - Rehabilitation supervision
    - Home safety systems

    Examples:
    - High severity fall: severity="high", person_name="Alice", location="bedroom"
    - Medium severity: severity="medium", confidence=0.85
    """

    input: FallDetectionInput
    output: FallDetectionInput

