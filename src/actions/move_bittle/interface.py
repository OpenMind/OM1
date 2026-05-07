from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class BittleMoveAction(str, Enum):
    """
    Petoi Bittle gait, posture, and skill commands supported by the ASCII token protocol.
    """

    WALK_FORWARD = "walk forward"
    WALK_LEFT = "walk left"
    WALK_RIGHT = "walk right"
    WALK_BACKWARD = "walk backward"
    CRAWL_FORWARD = "crawl forward"
    CRAWL_LEFT = "crawl left"
    CRAWL_RIGHT = "crawl right"
    TROT_FORWARD = "trot forward"
    TROT_LEFT = "trot left"
    TROT_RIGHT = "trot right"
    STAND_STILL = "stand still"
    BALANCE = "balance"
    BUTT_UP = "butt up"
    CHECK_AROUND = "check around"
    STRETCH = "stretch"
    GREETING = "greeting"
    PEE_POSE = "pee pose"
    PUSH_UP = "push up"
    REST = "rest"
    STEP_IN_PLACE = "step in place"
    BACK_FLIP = "back flip"
    SIT = "sit"
    BUNNY_JUMP = "bunny jump"
    VIBRATE = "vibrate"


@dataclass
class BittleMoveInput:
    """
    Input interface for Petoi Bittle movement.

    Parameters
    ----------
    action : BittleMoveAction
        The gait, posture, or skill command to execute on the Bittle robot. Supported actions are walk forward,
        walk left, walk right, walk backward, crawl forward, crawl left, crawl right, trot forward, trot left,
        trot right, stand still, balance, butt up, check around, stretch, greeting, pee pose, push up, rest,
        step in place, back flip, sit, bunny jump, and vibrate.
    """

    action: BittleMoveAction


@dataclass
class BittleMove(Interface[BittleMoveInput, BittleMoveInput]):
    """
    Move or pose a Petoi Bittle robot using documented Petoi ASCII gait, posture, and skill tokens.
    """

    input: BittleMoveInput
    output: BittleMoveInput
