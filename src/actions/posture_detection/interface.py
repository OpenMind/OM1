from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class PostureType(str, Enum):
    """
    Enumeration of posture types that can be detected.
    """

    GOOD = "good"  # Healthy posture
    SLUMPED = "slumped"  # Slouching forward
    LEANING = "leaning"  # Leaning to one side
    HUNCHED = "hunched"  # Rounded shoulders, forward head
    ASYMMETRIC = "asymmetric"  # Uneven posture
    LAYING = "laying"  # Person is laying down (may indicate fatigue)


class PostureSeverity(str, Enum):
    """
    Enumeration of posture issue severity levels.
    """

    MILD = "mild"  # Minor issue, gentle reminder
    MODERATE = "moderate"  # Noticeable issue, stronger reminder
    SEVERE = "severe"  # Serious issue, urgent reminder


@dataclass
class PostureDetectionInput:
    """
    Input interface for the PostureDetection action.

    Parameters
    ----------
    posture_type : PostureType
        The type of posture detected.
    severity : PostureSeverity
        The severity level of the posture issue.
    duration_minutes : float, optional
        How long the person has been in this posture (in minutes).
    person_name : str, optional
        Name of the person (if known from face recognition).
    recommendation : str, optional
        Recommended action to improve posture.
    """

    posture_type: PostureType
    severity: PostureSeverity
    duration_minutes: float = 0.0
    person_name: str = ""
    recommendation: str = ""


@dataclass
class PostureDetection(Interface[PostureDetectionInput, PostureDetectionInput]):
    """
    This action allows the robot to detect and remind users about their posture.

    The robot uses computer vision and pose estimation to monitor human posture
    in real-time. When poor posture is detected, the robot can:
    - Provide gentle reminders to adjust posture
    - Suggest exercises or stretches
    - Track posture patterns over time
    - Provide ergonomic recommendations

    This is particularly useful for:
    - Office workers who sit for long periods
    - Students studying at desks
    - People with back or neck pain
    - Rehabilitation and physical therapy

    Examples:
    - Slumped posture: posture_type="slumped", severity="moderate", duration_minutes=30
    - Good posture reminder: posture_type="good", severity="mild"
    """

    input: PostureDetectionInput
    output: PostureDetectionInput

