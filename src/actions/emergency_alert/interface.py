from dataclasses import dataclass

from actions.base import Interface


@dataclass
class EmergencyAlertInput:
    action: str


@dataclass
class EmergencyAlert(Interface[EmergencyAlertInput, EmergencyAlertInput]):
    """
    This action allows you to speak an emergency alert.
    """

    input: EmergencyAlertInput
    output: EmergencyAlertInput
