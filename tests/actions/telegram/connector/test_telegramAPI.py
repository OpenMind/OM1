"""Tests for the Telegram API connector."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock modules at module load time BEFORE any other imports
mock_zenoh = MagicMock()
mock_zenoh_msgs = MagicMock()
sys.modules["zenoh"] = mock_zenoh
sys.modules["zenoh_msgs"] = mock_zenoh_msgs

from actions.telegram.connector.telegramAPI import (  # noqa: E402
    TelegramAPIConfig,
    TelegramAPIConnector,
)
from actions.telegram.interface import TelegramInput  # noqa: E402


@pytest.fixture
def valid_config():
    """Create a valid config for testing."""
    return TelegramAPIConfig(bot_token="test_token_123", chat_id="123456789")


@pytest.fixture
def empty_token_config():
    """Create a config with empty bot token."""
    return TelegramAPIConfig(bot_token="", chat_id="123456789")


@pytest.fixture
def empty_chat_id_config():
    """Create a config with empty chat ID."""
    return TelegramAPIConfig(bot_token="test_token_123", chat_id="")


@pytest.fixture
def telegram_input():
    """Create a TelegramInput instance."""
    return TelegramInput(action="Hello from robot!")


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_zenoh.reset_mock()
    mock_zenoh_msgs.reset_mock()
    yield


class TestTelegramAPIConfig:
    """Test the Telegram API configuration class."""

    def test_valid_config(self):
        """Test creating config with valid values."""
        config = TelegramAPIConfig(bot_token="my_token", chat_id="12345")
        assert config.bot_token == "my_token"
        assert config.chat_id == "12345"


class TestTelegramAPIConnector:
    """Test the Telegram API connector."""

    def test_init_with_valid_config(self, valid_config):
        """Test initialization with valid configuration."""
        connector = TelegramAPIConnector(valid_config)
        assert connector.config.bot_token == "test_token_123"
        assert connector.config.chat_id == "123456789"

    def test_init_with_empty_token_logs_warning(self, empty_token_config):
        """Test initialization with empty token logs warning."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            TelegramAPIConnector(empty_token_config)
            mock_logging.warning.assert_called()

    def test_init_with_empty_chat_id_logs_warning(self, empty_chat_id_config):
        """Test initialization with empty chat ID logs warning."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            TelegramAPIConnector(empty_chat_id_config)
            mock_logging.warning.assert_called()

    @pytest.mark.asyncio
    async def test_connect_without_credentials(
        self, empty_token_config, telegram_input
    ):
        """Test connect when credentials not configured."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            connector = TelegramAPIConnector(empty_token_config)
            await connector.connect(telegram_input)
            mock_logging.error.assert_called_with("Telegram credentials not configured")

    @pytest.mark.asyncio
    async def test_connect_success(self, valid_config, telegram_input):
        """Test connect with successful API response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": {"message_id": 12345}})

        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession",
            return_value=mock_session_context,
        ):
            connector = TelegramAPIConnector(valid_config)
            await connector.connect(telegram_input)

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert "api.telegram.org" in call_args[0][0]
            assert call_args[1]["json"]["text"] == "Hello from robot!"
            assert call_args[1]["json"]["chat_id"] == "123456789"

    @pytest.mark.asyncio
    async def test_connect_api_error(self, valid_config, telegram_input):
        """Test connect with API error response."""
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession",
            return_value=mock_session_context,
        ):
            with patch(
                "actions.telegram.connector.telegramAPI.logging"
            ) as mock_logging:
                connector = TelegramAPIConnector(valid_config)
                await connector.connect(telegram_input)

                mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_connect_exception(self, valid_config, telegram_input):
        """Test connect handles exception."""
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.side_effect = Exception("Network error")

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession",
            return_value=mock_session_context,
        ):
            connector = TelegramAPIConnector(valid_config)

            with pytest.raises(Exception, match="Network error"):
                await connector.connect(telegram_input)
