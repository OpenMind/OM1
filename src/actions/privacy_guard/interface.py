"""Privacy Guard action interface.

Defines the action vocabulary the LLM can use to protect human privacy.
The cortex selects one of these actions each tick based on VLM captions.
"""

from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class PrivacyAction(str, Enum):
    """Available privacy-response actions."""

    LOOK_AWAY = "look_away"
    RESUME = "resume"
    STOP_STREAM = "stop_stream"
    IDLE = "idle"


@dataclass
class PrivacyGuardInput:
    """Input parameters for the privacy guard action.

    Parameters
    ----------
    action : PrivacyAction
        The privacy action to execute.
    """

    action: PrivacyAction


@dataclass
class PrivacyGuard(Interface[PrivacyGuardInput, PrivacyGuardInput]):
    """This action controls the robot's privacy response.
    Use 'look_away' when a privacy breach is detected (screens, documents, typing, private areas).
    Use 'resume' only when the environment is confirmed clear of all privacy threats.
    Use 'stop_stream' to kill the video feed for critical breaches.
    Use 'idle' when the environment is secure and no action is needed."""

    input: PrivacyGuardInput
    output: PrivacyGuardInput
