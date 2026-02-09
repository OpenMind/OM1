import logging
import os
import warnings

from dotenv import load_dotenv

from actions.base import ActionConfig, ActionConnector
from actions.tweet.interface import TweetInput


class TweetAPIConnector(ActionConnector[ActionConfig, TweetInput]):
    """
    Connector for Twitter API.

    This connector integrates with Twitter API v2 to post tweets from the robot.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the Twitter API connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        load_dotenv()

        # Suppress tweepy warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            import tweepy  # type: ignore

            self.client = tweepy.Client(
                consumer_key=os.getenv("TWITTER_API_KEY"),
                consumer_secret=os.getenv("TWITTER_API_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            )

    async def connect(self, output_interface: TweetInput) -> None:
        """
        Send tweet via Twitter API.

        Parameters
        ----------
        output_interface : TweetInput
            The TweetInput interface containing the tweet text.
        """
        try:
            # Log the tweet we're about to send
            # FIXED: Changed from output_interface.tweet to output_interface.action
            tweet_to_make = {"action": output_interface.action}
            logging.info(f"SendThisToTwitterAPI: {tweet_to_make}")

            # Send tweet
            # FIXED: Changed from output_interface.tweet to output_interface.action
            response = self.client.create_tweet(text=output_interface.action)
            tweet_id = response.data["id"]
            tweet_url = f"https://twitter.com/user/status/{tweet_id}"
            logging.info(f"Tweet sent successfully! URL: {tweet_url}")

        except Exception as e:
            logging.error(f"Failed to send tweet: {str(e)}")
            raise
