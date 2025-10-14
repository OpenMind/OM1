# src/actions/selfie/interface.py
from dataclasses import dataclass
from actions.base import Interface

@dataclass
class SelfieInput:
    """
    Arguments for taking a selfie and enrolling it to the gallery.

    action:      The person ID (e.g., "wendy"). Will create/update gallery/<id>.
    timeout_sec: Seconds to wait for exactly one face (default 15).
    """
    action: str
    timeout_sec: int = 15

@dataclass
class Selfie(Interface[SelfieInput, SelfieInput]):
    """
    This action takes a selfie from the live camera and enrolls it to the face gallery.
    """
    input: SelfieInput
    output: SelfieInput
