# tests/actions/tweet/test_twitter_api_connector.py
"""Unit tests for the Tweet action connector."""

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.tweet.interface import TweetInput

# Mock tweepy before importing the connector
sys.modules["tweepy"] = MagicMock()


class TestTweetAPIConnector:
    """Tests for TweetAPIConnector."""

    @pytest.fixture
    def mock_tweepy_client(self):
        """Create a mock tweepy Client."""
        mock_client = MagicMock()
        return mock_client

    def test_connector_initialization(self, mock_tweepy_client):
        """Test connector initialization and environment loading."""
        with patch("tweepy.Client", return_value=mock_tweepy_client):
            with patch.dict(
                os.environ,
                {
                    "TWITTER_API_KEY": "key",
                    "TWITTER_API_SECRET": "secret",
                    "TWITTER_ACCESS_TOKEN": "token",
                    "TWITTER_ACCESS_TOKEN_SECRET": "token_secret",
                },
            ):
                from actions.base import ActionConfig
                from actions.tweet.connector.twitterAPI import TweetAPIConnector

                config = ActionConfig()
                connector = TweetAPIConnector(config)

                # Verify Client was initialized with correct env variables
                from tweepy import Client

                Client.assert_called_with(
                    consumer_key="key",
                    consumer_secret="secret",
                    access_token="token",
                    access_token_secret="token_secret",
                )
                assert connector.client == mock_tweepy_client

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_tweepy_client, caplog):
        """Test successful tweet posting."""
        mock_response = MagicMock()
        mock_response.data = {"id": "12345"}
        mock_tweepy_client.create_tweet.return_value = mock_response

        with patch("tweepy.Client", return_value=mock_tweepy_client):
            from actions.base import ActionConfig
            from actions.tweet.connector.twitterAPI import TweetAPIConnector

            config = ActionConfig()
            connector = TweetAPIConnector(config)

            tweet_input = TweetInput(action="Hello, World! #OM1")

            with caplog.at_level(logging.INFO):
                await connector.connect(tweet_input)

            # Verify tweet was created
            mock_tweepy_client.create_tweet.assert_called_once_with(
                text="Hello, World! #OM1"
            )

            # Check success logs
            assert "Tweet sent successfully!" in caplog.text
            assert "URL: https://twitter.com/user/status/12345" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_failure_logs_error(self, mock_tweepy_client, caplog):
        """Test handles and logs tweet posting failure."""
        mock_tweepy_client.create_tweet.side_effect = Exception("API limit reached")

        with patch("tweepy.Client", return_value=mock_tweepy_client):
            from actions.base import ActionConfig
            from actions.tweet.connector.twitterAPI import TweetAPIConnector

            config = ActionConfig()
            connector = TweetAPIConnector(config)

            tweet_input = TweetInput(action="Trying to tweet during failure")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(Exception, match="API limit reached"):
                    await connector.connect(tweet_input)

            # Check failure logs
            assert "Failed to send tweet: API limit reached" in caplog.text

    def test_connector_inherits_from_action_connector(self):
        """Test that TweetAPIConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.tweet.connector.twitterAPI import TweetAPIConnector

        assert issubclass(TweetAPIConnector, ActionConnector)
