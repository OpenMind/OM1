"""
Emergency Call Action Interface

Provides interfaces for emergency call actions with support for
multi-modal triggers and tiered response escalation.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

from actions.base import Interface


class EmergencyTriggerType(str, Enum):
    """
    Types of triggers that can activate emergency response.
    """

    VOICE_KEYWORD = "voice_keyword"
    FALL_DETECTION = "fall_detection"
    PHYSICAL_BUTTON = "physical_button"
    MANUAL = "manual"
    HEART_RATE_ALERT = "heart_rate_alert"


class EmergencyLevel(IntEnum):
    """
    Emergency severity levels determining response escalation.

    LOW: Non-life-threatening, notification only
    MEDIUM: Potentially serious, escalate to phone call
    HIGH: Serious, escalate to emergency services
    CRITICAL: Life-threatening, immediate emergency response
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EmergencyResponseStatus(str, Enum):
    """
    Status of emergency response.
    """

    DETECTED = "detected"
    NOTIFICATION_SENT = "notification_sent"
    CALL_INITIATED = "call_initiated"
    EMERGENCY_DISPATCHED = "emergency_dispatched"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class EmergencyCallInput:
    """
    Input interface for the EmergencyCall action.

    Parameters
    ----------
    trigger_type : EmergencyTriggerType
        What triggered the emergency (voice, fall, button, etc.)
    emergency_level : EmergencyLevel
        Severity level determining escalation path
    location : str
        Where the emergency occurred
    user_message : Optional[str]
        Message from user or description of situation
    user_id : Optional[str]
        ID of the user in distress
    sensor_data : Optional[dict]
        Additional sensor data (IMU readings, heart rate, etc.)
    timestamp : Optional[str]
        ISO timestamp of when emergency was detected
    """

    trigger_type: EmergencyTriggerType
    emergency_level: EmergencyLevel
    location: str
    user_message: Optional[str] = None
    user_id: Optional[str] = None
    sensor_data: Optional[dict] = None
    timestamp: Optional[str] = None


@dataclass
class EmergencyCall(Interface[EmergencyCallInput, EmergencyCallInput]):
    """
    Emergency Call action for initiating emergency response.

    This action allows the robot to initiate a tiered emergency response:
    1. Send notifications to family members
    2. Initiate phone calls
    3. Contact emergency services

    The response level depends on the emergency_level field.

    Example:
        EmergencyCall(
            input=EmergencyCallInput(
                trigger_type=EmergencyTriggerType.FALL_DETECTION,
                emergency_level=EmergencyLevel.HIGH,
                location="kitchen",
                user_message="Fall detected by IMU"
            ),
            output=EmergencyCallInput(...)
        )
    """

    input: EmergencyCallInput
    output: EmergencyCallInput
