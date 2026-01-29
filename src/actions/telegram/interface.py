from dataclasses import dataclass

from actions.base import Interface


@dataclass
class TelegramInput:
    """
    Input interface for the Telegram Message action.

    Parameters
    ----------
    action : str
        The text content to be sent as a message to Telegram.
        Can include emojis and basic formatting.
    """

    action: str = ""


@dataclass
class Telegram(Interface[TelegramInput, TelegramInput]):
    """
    This action allows the robot to send messages to Telegram.

    Effect: Sends the specified text content as a message to the configured
    Telegram chat using the Telegram Bot API. The message is sent immediately
    and logged upon successful delivery.
    """

    input: TelegramInput
    output: TelegramInput
