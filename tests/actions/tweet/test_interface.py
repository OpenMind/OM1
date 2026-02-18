"""Tests for the Tweet action interface."""

from actions.tweet.interface import Tweet, TweetInput


class TestTweetInput:
    """Tests for the TweetInput dataclass."""

    def test_tweet_input_default(self):
        """Test creating TweetInput with default value."""
        tweet_input = TweetInput()
        assert tweet_input.action == ""

    def test_tweet_input_with_message(self):
        """Test creating TweetInput with message."""
        tweet_input = TweetInput(action="Hello from robot!")
        assert tweet_input.action == "Hello from robot!"

    def test_tweet_input_max_length(self):
        """Test creating TweetInput with 280 character limit."""
        long_tweet = "A" * 280
        tweet_input = TweetInput(action=long_tweet)
        assert len(tweet_input.action) == 280


class TestTweet:
    """Tests for the Tweet interface."""

    def test_tweet_creation(self):
        """Test creating Tweet with input and output."""
        tweet_input = TweetInput(action="Test tweet")
        tweet = Tweet(input=tweet_input, output=tweet_input)
        assert tweet.input == tweet_input
        assert tweet.output == tweet_input

    def test_tweet_different_input_output(self):
        """Test creating Tweet with different input and output."""
        input_tweet = TweetInput(action="Input tweet")
        output_tweet = TweetInput(action="Output tweet")
        tweet = Tweet(input=input_tweet, output=output_tweet)
        assert tweet.input.action == "Input tweet"
        assert tweet.output.action == "Output tweet"
