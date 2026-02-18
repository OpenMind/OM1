"""
Emergency Call Plugin for OM1

A comprehensive emergency response system with multi-modal triggers
and tiered response escalation.

Features:
- Multi-modal triggers: voice, IMU fall detection, physical buttons
- Tiered response: notification → call → emergency center
- Privacy: encrypted logs with auto-deletion

Example usage:
    EmergencyCallInput(
        trigger_type=EmergencyTriggerType.VOICE_KEYWORD,
        emergency_level=EmergencyLevel.CRITICAL,
        location="living_room",
        user_message="I need help, I fell!"
    )
"""

from actions.emergency_call.connector.emergency_call_connector import (
    EmergencyCallConfig,
    EmergencyCallConnector,
)
from actions.emergency_call.interface import (
    EmergencyCall,
    EmergencyCallInput,
    EmergencyLevel,
    EmergencyResponseStatus,
    EmergencyTriggerType,
)

__all__ = [
    "EmergencyCall",
    "EmergencyCallConfig",
    "EmergencyCallConnector",
    "EmergencyCallInput",
    "EmergencyLevel",
    "EmergencyResponseStatus",
    "EmergencyTriggerType",
]
