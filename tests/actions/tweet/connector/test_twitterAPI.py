"""Tests for the Twitter API connector."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock modules at module load time BEFORE any other imports
mock_tweepy = MagicMock()
mock_zenoh = MagicMock()
mock_zenoh_msgs = MagicMock()
sys.modules["tweepy"] = mock_tweepy
sys.modules["zenoh"] = mock_zenoh
sys.modules["zenoh_msgs"] = mock_zenoh_msgs

from actions.base import ActionConfig  # noqa: E402
from actions.tweet.connector.twitterAPI import TweetAPIConnector  # noqa: E402
from actions.tweet.interface import TweetInput  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return ActionConfig()


@pytest.fixture
def tweet_input():
    """Create a TweetInput instance."""
    return TweetInput(action="Hello from robot!")


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_tweepy.reset_mock()
    mock_zenoh.reset_mock()
    mock_zenoh_msgs.reset_mock()
    yield


class TestTweetAPIConnector:
    """Test the Twitter API connector."""

    @patch.dict(
        "os.environ",
        {
            "TWITTER_API_KEY": "test_key",
            "TWITTER_API_SECRET": "test_secret",
            "TWITTER_ACCESS_TOKEN": "test_token",
            "TWITTER_ACCESS_TOKEN_SECRET": "test_token_secret",
        },
    )
    def test_init(self, default_config):
        """Test initialization of TweetAPIConnector."""
        mock_client = Mock()
        mock_tweepy.Client.return_value = mock_client

        connector = TweetAPIConnector(default_config)

        assert connector.client is not None
        mock_tweepy.Client.assert_called_once()

    @patch.dict(
        "os.environ",
        {
            "TWITTER_API_KEY": "test_key",
            "TWITTER_API_SECRET": "test_secret",
            "TWITTER_ACCESS_TOKEN": "test_token",
            "TWITTER_ACCESS_TOKEN_SECRET": "test_token_secret",
        },
    )
    @pytest.mark.asyncio
    async def test_connect_success(self, default_config, tweet_input):
        """Test connect with successful API response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {"id": "123456789"}
        mock_client.create_tweet.return_value = mock_response
        mock_tweepy.Client.return_value = mock_client

        connector = TweetAPIConnector(default_config)
        await connector.connect(tweet_input)

        mock_client.create_tweet.assert_called_once_with(text="Hello from robot!")

    @patch.dict(
        "os.environ",
        {
            "TWITTER_API_KEY": "test_key",
            "TWITTER_API_SECRET": "test_secret",
            "TWITTER_ACCESS_TOKEN": "test_token",
            "TWITTER_ACCESS_TOKEN_SECRET": "test_token_secret",
        },
    )
    @pytest.mark.asyncio
    async def test_connect_exception(self, default_config, tweet_input):
        """Test connect handles exception."""
        mock_client = Mock()
        mock_client.create_tweet.side_effect = Exception("API error")
        mock_tweepy.Client.return_value = mock_client

        connector = TweetAPIConnector(default_config)

        with pytest.raises(Exception, match="API error"):
            await connector.connect(tweet_input)

    @patch.dict(
        "os.environ",
        {
            "TWITTER_API_KEY": "test_key",
            "TWITTER_API_SECRET": "test_secret",
            "TWITTER_ACCESS_TOKEN": "test_token",
            "TWITTER_ACCESS_TOKEN_SECRET": "test_token_secret",
        },
    )
    @pytest.mark.asyncio
    async def test_connect_logs_tweet(self, default_config, tweet_input):
        """Test connect logs the tweet being sent."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = {"id": "123456789"}
        mock_client.create_tweet.return_value = mock_response
        mock_tweepy.Client.return_value = mock_client

        with patch("actions.tweet.connector.twitterAPI.logging") as mock_logging:
            connector = TweetAPIConnector(default_config)
            await connector.connect(tweet_input)

            mock_logging.info.assert_called()
