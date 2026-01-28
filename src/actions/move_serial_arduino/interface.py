from dataclasses import dataclass
from enum import Enum
from actions.base import Interface

class MovementAction(str, Enum):
    """
    Enumeration of possible movement actions.
    Expanded to support locomotion for Arduino robots.
    """
    BE_STILL = "be still"
    STOP = "stop"
    
    # Jumping (Original)
    JUMP_SMALL = "small jump"
    JUMP_MEDIUM = "medium jump"
    JUMP_BIG = "big jump"
    
    # Locomotion (New Additions)
    WALK = "walk"
    WALK_FORWARD = "walk forward"
    WALK_BACK = "walk back"
    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"
    DANCE = "dance"

@dataclass
class MoveInput:
    """
    Input interface for the Move action.
    """
    action: MovementAction

@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    A movement to be performed by the agent.
    Effect: Allows the agent to move via Arduino Serial.
    """
    input: MoveInput
    output: MoveInput