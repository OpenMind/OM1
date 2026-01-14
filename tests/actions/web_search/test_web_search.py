"""Tests for WebSearch action."""

from unittest.mock import MagicMock, patch

import pytest

from actions.web_search.connector.duckduckgo import (
    DuckDuckGoConfig,
    DuckDuckGoConnector,
)
from actions.web_search.interface import WebSearch, WebSearchInput, WebSearchOutput


class TestWebSearchInterface:
    """Tests for WebSearch interface."""

    def test_web_search_input_default(self):
        """Test WebSearchInput with default values."""
        input_obj = WebSearchInput()
        assert input_obj.action == ""

    def test_web_search_input_with_query(self):
        """Test WebSearchInput with a query."""
        input_obj = WebSearchInput(action="test query")
        assert input_obj.action == "test query"

    def test_web_search_output_default(self):
        """Test WebSearchOutput with default values."""
        output_obj = WebSearchOutput()
        assert output_obj.results == ""

    def test_web_search_output_with_results(self):
        """Test WebSearchOutput with results."""
        output_obj = WebSearchOutput(results="Search results here")
        assert output_obj.results == "Search results here"

    def test_web_search_interface(self):
        """Test WebSearch interface structure."""
        input_obj = WebSearchInput(action="test")
        output_obj = WebSearchOutput(results="results")
        web_search = WebSearch(input=input_obj, output=output_obj)
        assert web_search.input.action == "test"
        assert web_search.output.results == "results"


class TestDuckDuckGoConfig:
    """Tests for DuckDuckGoConfig."""

    def test_default_values(self):
        """Test config with default values."""
        config = DuckDuckGoConfig()
        assert config.max_results == 5
        assert config.region == "wt-wt"
        assert config.safesearch == "moderate"

    def test_custom_values(self):
        """Test config with custom values."""
        config = DuckDuckGoConfig(
            max_results=10,
            region="us-en",
            safesearch="off",
        )
        assert config.max_results == 10
        assert config.region == "us-en"
        assert config.safesearch == "off"


class TestDuckDuckGoConnector:
    """Tests for DuckDuckGoConnector."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("actions.web_search.connector.duckduckgo.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def connector(self, mock_io_provider):
        """Create a DuckDuckGoConnector instance."""
        config = DuckDuckGoConfig()
        conn = DuckDuckGoConnector(config)
        conn.io_provider = mock_io_provider
        return conn

    def test_init(self, mock_io_provider):
        """Test connector initialization."""
        config = DuckDuckGoConfig(max_results=3)
        connector = DuckDuckGoConnector(config)
        assert connector.config.max_results == 3
        assert connector._last_results is None

    def test_format_results_empty(self, connector):
        """Test formatting empty results."""
        result = connector._format_results([])
        assert result == "No results found."

    def test_format_results_with_data(self, connector):
        """Test formatting results with data."""
        results = [
            {
                "title": "Test Title",
                "body": "Test description",
                "href": "https://example.com",
            },
            {
                "title": "Another Title",
                "body": "Another description",
                "href": "https://example2.com",
            },
        ]
        formatted = connector._format_results(results)
        assert "Test Title" in formatted
        assert "Test description" in formatted
        assert "https://example.com" in formatted
        assert "Another Title" in formatted

    @pytest.mark.asyncio
    async def test_connect_empty_query(self, connector):
        """Test connect with empty query logs warning."""
        with patch(
            "actions.web_search.connector.duckduckgo.logging.warning"
        ) as mock_warning:
            input_obj = WebSearchInput(action="")
            await connector.connect(input_obj)
            mock_warning.assert_called_with("WebSearch: Empty search query provided")

    @pytest.mark.asyncio
    async def test_connect_success(self, connector, mock_io_provider):
        """Test successful web search."""
        mock_results = [
            {
                "title": "Test Result",
                "body": "Test body",
                "href": "https://test.com",
            }
        ]

        with patch("actions.web_search.connector.duckduckgo.DDGS") as mock_ddgs:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = iter(mock_results)
            mock_ddgs.return_value.__enter__ = MagicMock(
                return_value=mock_ddgs_instance
            )
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=None)

            input_obj = WebSearchInput(action="test query")
            await connector.connect(input_obj)

            assert connector._last_results is not None
            assert "Test Result" in connector._last_results
            mock_io_provider.add_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_search_error(self, connector):
        """Test handling of search error."""
        with patch("actions.web_search.connector.duckduckgo.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__ = MagicMock(
                side_effect=Exception("Network error")
            )

            with patch(
                "actions.web_search.connector.duckduckgo.logging.error"
            ) as mock_error:
                input_obj = WebSearchInput(action="test query")
                await connector.connect(input_obj)

                assert any(
                    "Search failed" in str(call) for call in mock_error.call_args_list
                )

    def test_get_last_results_none(self, connector):
        """Test get_last_results when no search has been done."""
        result = connector.get_last_results()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_results_after_search(self, connector):
        """Test get_last_results after a search."""
        mock_results = [
            {
                "title": "Result",
                "body": "Body",
                "href": "https://url.com",
            }
        ]

        with patch("actions.web_search.connector.duckduckgo.DDGS") as mock_ddgs:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.text.return_value = iter(mock_results)
            mock_ddgs.return_value.__enter__ = MagicMock(
                return_value=mock_ddgs_instance
            )
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=None)

            input_obj = WebSearchInput(action="test")
            await connector.connect(input_obj)

            result = connector.get_last_results()
            assert result is not None
            assert "Result" in result
