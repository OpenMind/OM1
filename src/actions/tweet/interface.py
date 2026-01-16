from dataclasses import dataclass

from actions.base import Interface


@dataclass
class TweetInput:
    """
    Input interface for the Tweet action.

    Parameters
    ----------
    action : str
        The text content to be posted as a tweet on Twitter/X.
        Limited to Twitter's character limits (280 characters for most accounts).
    """

    action: str = ""  # Make tweet optional with default empty string


@dataclass
class Tweet(Interface[TweetInput, TweetInput]):
    """
    This action allows the robot to post messages on Twitter/X.

    Effect: Publishes the specified text content as a tweet on the connected
    Twitter/X account using the Twitter API v2. The tweet is posted immediately
    and a tweet URL is logged upon successful posting.
    """

    input: TweetInput
    output: TweetInput
