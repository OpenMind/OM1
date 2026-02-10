import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.news import NewsConfig, NewsInput


class TestNewsConfig:
    """Tests for NewsConfig."""

    def test_default_values(self):
        """Test config with default values."""
        config = NewsConfig(api_key="test_api_key")
        assert config.api_key == "test_api_key"
        assert config.country == "us"
        assert config.category == "general"
        assert config.query == ""
        assert config.max_headlines == 5
        assert config.poll_interval == 900.0

    def test_custom_values(self):
        """Test config with custom values."""
        config = NewsConfig(
            api_key="my_key",
            country="gb",
            category="technology",
            query="AI",
            max_headlines=10,
            poll_interval=1800.0,
        )
        assert config.country == "gb"
        assert config.category == "technology"
        assert config.query == "AI"
        assert config.max_headlines == 10
        assert config.poll_interval == 1800.0


class TestNewsInput:
    """Tests for NewsInput."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.news.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def news_input(self, mock_io_provider):
        """Create a NewsInput instance."""
        config = NewsConfig(
            api_key="test_api_key",
            country="us",
            category="technology",
        )
        return NewsInput(config)

    def test_init_with_config(self, mock_io_provider):
        """Test initialization with config."""
        config = NewsConfig(api_key="test_key", country="tr", category="sports")
        news = NewsInput(config)
        assert news.api_key == "test_key"
        assert news.country == "tr"
        assert news.category == "sports"
        assert news.descriptor_for_LLM == "Current News"

    def test_init_without_api_key_logs_warning(self, mock_io_provider):
        """Test that missing API key logs a warning."""
        with patch("inputs.plugins.news.logging.warning") as mock_warning:
            config = NewsConfig(api_key="")
            NewsInput(config)
            mock_warning.assert_called_with("NewsInput: API key not provided")


class TestNewsInputFetch:
    """Tests for news fetching."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.news.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def news_input(self, mock_io_provider):
        """Create a NewsInput instance."""
        config = NewsConfig(api_key="test_api_key")
        return NewsInput(config)

    @pytest.mark.asyncio
    async def test_fetch_news_success(self, news_input):
        """Test successful news fetch."""
        mock_response_data = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": {"name": "TechCrunch"},
                    "title": "AI breakthrough announced",
                    "description": "Major AI development",
                },
                {
                    "source": {"name": "Wired"},
                    "title": "New tech trends",
                    "description": "Latest trends in tech",
                },
            ],
        }

        with patch("inputs.plugins.news.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await news_input._fetch_news()

            assert result is not None
            assert result["status"] == "ok"
            assert len(result["articles"]) == 2

    @pytest.mark.asyncio
    async def test_fetch_news_api_error(self, news_input):
        """Test handling of API error."""
        with patch("inputs.plugins.news.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Invalid API key")

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch("inputs.plugins.news.logging.error") as mock_error:
                result = await news_input._fetch_news()
                assert result is None
                assert any("401" in str(call) for call in mock_error.call_args_list)

    @pytest.mark.asyncio
    async def test_fetch_news_no_api_key(self, mock_io_provider):
        """Test that fetch returns None without API key."""
        config = NewsConfig(api_key="")
        news = NewsInput(config)
        result = await news._fetch_news()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_news_with_query(self, mock_io_provider):
        """Test fetch with search query."""
        config = NewsConfig(api_key="test_key", query="artificial intelligence")
        news = NewsInput(config)

        mock_response_data = {
            "status": "ok",
            "articles": [{"source": {"name": "Test"}, "title": "AI News"}],
        }

        with patch("inputs.plugins.news.aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await news._fetch_news()
            assert result is not None


class TestNewsInputPoll:
    """Tests for _poll behavior."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.news.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def news_input(self, mock_io_provider):
        """Create a NewsInput instance with short poll interval."""
        config = NewsConfig(api_key="test_api_key", poll_interval=10.0)
        return NewsInput(config)

    @pytest.mark.asyncio
    async def test_poll_returns_data_on_first_call(self, news_input):
        """Test that first poll fetches and returns data."""
        mock_data = {
            "status": "ok",
            "articles": [{"source": {"name": "BBC"}, "title": "News"}],
        }
        with patch.object(
            news_input, "_fetch_news", new_callable=AsyncMock, return_value=mock_data
        ):
            result = await news_input._poll()
            assert result is not None
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_poll_returns_none_before_interval(self, news_input):
        """Test that poll returns None when interval has not elapsed."""
        mock_data = {
            "status": "ok",
            "articles": [{"source": {"name": "BBC"}, "title": "News"}],
        }
        with patch.object(
            news_input, "_fetch_news", new_callable=AsyncMock, return_value=mock_data
        ):
            await news_input._poll()
            result = await news_input._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_poll_fetches_again_after_interval(self, news_input):
        """Test that poll fetches fresh data after interval elapses."""
        mock_data = {
            "status": "ok",
            "articles": [{"source": {"name": "BBC"}, "title": "Fresh news"}],
        }
        with patch.object(
            news_input, "_fetch_news", new_callable=AsyncMock, return_value=mock_data
        ):
            await news_input._poll()

            news_input._last_poll_time = time.time() - 20.0

            result = await news_input._poll()
            assert result is not None
            assert news_input._fetch_news.await_count == 2

    @pytest.mark.asyncio
    async def test_poll_does_not_send_duplicate_news(self, news_input):
        """Test that same news is not sent to LLM repeatedly between intervals."""
        mock_data = {
            "status": "ok",
            "articles": [{"source": {"name": "BBC"}, "title": "Breaking"}],
        }
        with patch.object(
            news_input, "_fetch_news", new_callable=AsyncMock, return_value=mock_data
        ):
            result1 = await news_input._poll()
            await news_input.raw_to_text(result1)
            assert len(news_input.messages) == 1

            result2 = await news_input._poll()
            await news_input.raw_to_text(result2)
            assert len(news_input.messages) == 1


class TestNewsInputRawToText:
    """Tests for raw_to_text conversion."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.news.IOProvider") as mock:
            yield mock

    @pytest.fixture
    def news_input(self, mock_io_provider):
        """Create a NewsInput instance."""
        config = NewsConfig(api_key="test_api_key")
        return NewsInput(config)

    @pytest.mark.asyncio
    async def test_raw_to_text_success(self, news_input):
        """Test successful conversion to text."""
        raw_data = {
            "status": "ok",
            "articles": [
                {"source": {"name": "BBC"}, "title": "Breaking news story"},
                {"source": {"name": "CNN"}, "title": "Another headline"},
            ],
        }

        result = await news_input._raw_to_text(raw_data)

        assert result is not None
        assert "Breaking news story" in result.message
        assert "BBC" in result.message
        assert "Another headline" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, news_input):
        """Test that None input returns None."""
        result = await news_input._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_empty_articles(self, news_input):
        """Test handling of empty articles list."""
        raw_data = {"status": "ok", "articles": []}

        result = await news_input._raw_to_text(raw_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_filters_removed_articles(self, news_input):
        """Test that [Removed] articles are filtered out."""
        raw_data = {
            "status": "ok",
            "articles": [
                {"source": {"name": "Test"}, "title": "[Removed]"},
                {"source": {"name": "BBC"}, "title": "Valid headline"},
            ],
        }

        result = await news_input._raw_to_text(raw_data)

        assert result is not None
        assert "[Removed]" not in result.message
        assert "Valid headline" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_respects_max_headlines(self, mock_io_provider):
        """Test that max_headlines is respected."""
        config = NewsConfig(api_key="test_key", max_headlines=2)
        news = NewsInput(config)

        raw_data = {
            "status": "ok",
            "articles": [
                {"source": {"name": "A"}, "title": "First"},
                {"source": {"name": "B"}, "title": "Second"},
                {"source": {"name": "C"}, "title": "Third"},
            ],
        }

        result = await news._raw_to_text(raw_data)

        assert result is not None
        assert "First" in result.message
        assert "Second" in result.message
        assert "Third" not in result.message


class TestNewsInputFormatted:
    """Tests for formatted_latest_buffer."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.news.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def news_input(self, mock_io_provider):
        """Create a NewsInput instance."""
        config = NewsConfig(api_key="test_api_key")
        news = NewsInput(config)
        news.io_provider = mock_io_provider
        return news

    def test_formatted_latest_buffer_empty(self, news_input):
        """Test that empty buffer returns None."""
        result = news_input.formatted_latest_buffer()
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_with_message(self, news_input):
        """Test formatting with message in buffer."""
        raw_data = {
            "status": "ok",
            "articles": [
                {"source": {"name": "Reuters"}, "title": "Market update"},
            ],
        }

        await news_input.raw_to_text(raw_data)
        result = news_input.formatted_latest_buffer()

        assert result is not None
        assert "Current News" in result
        assert "Market update" in result
        assert "// START" in result
        assert "// END" in result

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_clears_messages(self, news_input):
        """Test that buffer is cleared after formatting."""
        raw_data = {
            "status": "ok",
            "articles": [
                {"source": {"name": "AP"}, "title": "Test headline"},
            ],
        }

        await news_input.raw_to_text(raw_data)
        assert len(news_input.messages) == 1

        news_input.formatted_latest_buffer()
        assert len(news_input.messages) == 0
