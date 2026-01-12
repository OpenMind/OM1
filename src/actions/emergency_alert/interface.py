from dataclasses import dataclass

from actions.base import Interface


@dataclass
class EmergencyAlertInput:
    """
    Input interface for the EmergencyAlert action.

    Parameters
    ----------
    action : str
        The emergency message to be announced immediately through text-to-speech.
        Should contain a clear description of the emergency situation.
    """

    action: str


@dataclass
class EmergencyAlert(Interface[EmergencyAlertInput, EmergencyAlertInput]):
    """
    This action allows the robot to broadcast emergency alert messages.

    Effect: Immediately announces critical information through text-to-speech.
    Designed for urgent situations requiring immediate attention (e.g., security alerts,
    injury detection, distress calls). Unlike regular speak actions, emergency alerts
    skip message queuing and cannot be interrupted, ensuring critical messages are
    delivered without delay.

    Primary use case: Security/guard mode when detecting unknown people, injuries,
    or situations requiring immediate human intervention.
    """

    input: EmergencyAlertInput
    output: EmergencyAlertInput
