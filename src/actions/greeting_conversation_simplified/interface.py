from dataclasses import dataclass

from actions.base import Interface


@dataclass
class GreetingConversationSimplifiedInput:
    """
    Input interface for the simplified GreetingConversation action.

    Parameters
    ----------
    response : str
        The spoken answer to the user.
    """

    response: str


@dataclass
class GreetingConversationSimplified(
    Interface[GreetingConversationSimplifiedInput, GreetingConversationSimplifiedInput]
):
    """
    Respond to the user. Put your spoken answer in 'response'.
    """

    input: GreetingConversationSimplifiedInput
    output: GreetingConversationSimplifiedInput
