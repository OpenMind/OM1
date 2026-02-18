from dataclasses import dataclass

from actions.base import Interface


@dataclass
class DiscordInput:
    """
    Input interface for the Discord Webhook action.

    Parameters
    ----------
    action : str
        The text content to be sent as a message to Discord.
        Can include markdown formatting supported by Discord.
    """

    action: str = ""


@dataclass
class Discord(Interface[DiscordInput, DiscordInput]):
    """
    This action allows the robot to send messages to Discord via webhooks.

    Effect: Sends the specified text content as a message to the configured
    Discord channel using a webhook URL. The message is sent immediately
    and logged upon successful delivery.
    """

    input: DiscordInput
    output: DiscordInput
