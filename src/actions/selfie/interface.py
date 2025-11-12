# src/actions/selfie/interface.py
from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SelfieInput:
    """
    Input to enroll a selfie through the face HTTP service.

    Parameters
    ----------
    action : str
        The person ID (e.g., "wendy"). Will create/update gallery/<id>.
    timeout_sec : int, optional
        Seconds to wait for exactly one face (default 15).
    """

    action: str
    timeout_sec: int = 5


@dataclass
class Selfie(Interface[SelfieInput, SelfieInput]):
    """
    This action takes a selfie from the live camera and enrolls it to the face gallery.
    """

    input: SelfieInput
    output: SelfieInput
