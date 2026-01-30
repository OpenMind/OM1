from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class ConversationState(Enum):
    """
    States for greeting conversation.
    """

    CONVERSING = "conversing"
    CONCLUDING = "concluding"
    FINISHED = "finished"


@dataclass
class GreetingConversationInput:
    """
    Input interface for the GreetingConversation action.

    Parameters
    ----------
    response : str
        The speech output from the robot.
    conversation_state: ConversationState
        The current state of the greeting conversation.
    confidence : float
        Confidence score of the conversation state recognition from 0.0 to 1.0.
    speech_clarity : float
        User's voice input clarity score from 0.0 to 1.0.
    """

    response: str
    conversation_state: ConversationState
    confidence: float
    speech_clarity: float


@dataclass
class GreetingConversation(
    Interface[GreetingConversationInput, GreetingConversationInput]
):
    """
    This action manages greeting conversations for the robot.

    The robot responds to user greetings based on confidence and speech clarity scores.
    - response: The text the robot will say in response.
    - conversation_state: The current state of the greeting conversation.
    - confidence: Confidence score of the conversation state recognition from 0.0 to 1.0.
    - speech_clarity: User's voice input clarity score from 0.0 to 1.0.
    """

    input: GreetingConversationInput
    output: GreetingConversationInput
