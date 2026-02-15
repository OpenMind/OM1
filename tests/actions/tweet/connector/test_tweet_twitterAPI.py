from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.base import ActionConfig
from actions.tweet.connector.twitterAPI import TweetAPIConnector
from actions.tweet.interface import TweetInput

_mock_tweepy = MagicMock()


@pytest.fixture
def mock_env_vars():
    """Mock Twitter API environment variables."""
    env = {
        "TWITTER_API_KEY": "test_api_key",
        "TWITTER_API_SECRET": "test_api_secret",
        "TWITTER_ACCESS_TOKEN": "test_access_token",
        "TWITTER_ACCESS_TOKEN_SECRET": "test_access_token_secret",
    }
    with patch.dict("os.environ", env):
        yield env


@pytest.fixture
def twitter_connector(mock_env_vars):
    """Create TweetAPIConnector with mocked dependencies."""
    with (
        patch.dict("sys.modules", {"tweepy": _mock_tweepy}),
        patch("actions.tweet.connector.twitterAPI.load_dotenv"),
    ):
        mock_client = Mock()
        _mock_tweepy.Client.return_value = mock_client

        connector = TweetAPIConnector(ActionConfig())
        connector.client = mock_client
        return connector


class TestTweetAPIConnectorInit:
    """Test TweetAPIConnector initialization."""

    def test_init_creates_tweepy_client(self, mock_env_vars):
        """Test that init creates a tweepy Client with env vars."""
        with (
            patch.dict("sys.modules", {"tweepy": _mock_tweepy}),
            patch("actions.tweet.connector.twitterAPI.load_dotenv"),
        ):
            mock_client = Mock()
            _mock_tweepy.Client.return_value = mock_client

            connector = TweetAPIConnector(ActionConfig())
            assert connector.client == mock_client


class TestTweetAPIConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_sends_tweet(self, twitter_connector):
        """Test that connect sends a tweet via Twitter API."""
        twitter_connector.client.create_tweet.return_value = Mock(data={"id": "12345"})

        tweet_input = TweetInput(action="Hello world!")
        with patch("actions.tweet.connector.twitterAPI.logging") as mock_logging:
            await twitter_connector.connect(tweet_input)

            twitter_connector.client.create_tweet.assert_called_once_with(
                text="Hello world!"
            )
            mock_logging.info.assert_any_call(
                "SendThisToTwitterAPI: {'action': 'Hello world!'}"
            )
            mock_logging.info.assert_any_call(
                "Tweet sent successfully! URL: https://twitter.com/user/status/12345"
            )

    @pytest.mark.asyncio
    async def test_connect_empty_tweet(self, twitter_connector):
        """Test sending an empty tweet."""
        twitter_connector.client.create_tweet.return_value = Mock(data={"id": "99999"})

        tweet_input = TweetInput(action="")
        await twitter_connector.connect(tweet_input)

        twitter_connector.client.create_tweet.assert_called_once_with(text="")

    @pytest.mark.asyncio
    async def test_connect_api_error_raises(self, twitter_connector):
        """Test that API errors are re-raised after logging."""
        twitter_connector.client.create_tweet.side_effect = Exception("API Error")

        tweet_input = TweetInput(action="Test tweet")
        with (
            patch("actions.tweet.connector.twitterAPI.logging") as mock_logging,
            pytest.raises(Exception, match="API Error"),
        ):
            await twitter_connector.connect(tweet_input)
            mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_connect_logs_tweet_content(self, twitter_connector):
        """Test that the tweet content is logged before sending."""
        twitter_connector.client.create_tweet.return_value = Mock(data={"id": "11111"})

        tweet_input = TweetInput(action="Logging test")
        with patch("actions.tweet.connector.twitterAPI.logging") as mock_logging:
            await twitter_connector.connect(tweet_input)
            mock_logging.info.assert_any_call(
                "SendThisToTwitterAPI: {'action': 'Logging test'}"
            )
