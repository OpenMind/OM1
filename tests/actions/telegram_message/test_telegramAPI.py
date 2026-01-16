from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.telegram.connector.telegramAPI import (
    TelegramAPIConfig,
    TelegramAPIConnector,
)
from actions.telegram.interface import Telegram, TelegramInput


def test_telegram_input_default():
    """Test TelegramInput with default values."""
    input_obj = TelegramInput()
    assert input_obj.action == ""


def test_telegram_input_with_value():
    """Test TelegramInput with custom value."""
    input_obj = TelegramInput(action="Hello from robot!")
    assert input_obj.action == "Hello from robot!"


def test_telegram_input_with_emoji():
    """Test TelegramInput with emoji."""
    input_obj = TelegramInput(action="Battery low! Please charge.")
    assert "Battery" in input_obj.action


def test_telegram_interface():
    """Test Telegram interface creation."""
    input_obj = TelegramInput(action="Test message")
    output_obj = TelegramInput(action="Test message")
    message = Telegram(input=input_obj, output=output_obj)
    assert message.input.action == "Test message"
    assert message.output.action == "Test message"


def test_init_with_credentials():
    """Test initialization with credentials."""
    config = TelegramAPIConfig(bot_token="test-bot-token", chat_id="test-chat-id")
    connector = TelegramAPIConnector(config)
    assert connector.config.bot_token == "test-bot-token"
    assert connector.config.chat_id == "test-chat-id"


def test_init_without_bot_token():
    """Test initialization without bot token logs warning."""
    with patch(
        "actions.telegram.connector.telegramAPI.logging.warning"
    ) as mock_warning:
        config = TelegramAPIConfig(bot_token="", chat_id="test-chat-id")
        connector = TelegramAPIConnector(config)
        assert connector.config.bot_token == ""
        mock_warning.assert_any_call("Telegram Bot Token not provided in configuration")


def test_init_without_chat_id():
    """Test initialization without chat id logs warning."""
    with patch(
        "actions.telegram.connector.telegramAPI.logging.warning"
    ) as mock_warning:
        config = TelegramAPIConfig(bot_token="test-token", chat_id="")
        connector = TelegramAPIConnector(config)
        assert connector.config.chat_id == ""
        mock_warning.assert_any_call("Telegram Chat ID not provided in configuration")


def test_init_without_any_credentials():
    """Test initialization without any credentials logs both warnings."""
    with patch(
        "actions.telegram.connector.telegramAPI.logging.warning"
    ) as mock_warning:
        config = TelegramAPIConfig(bot_token="", chat_id="")
        connector = TelegramAPIConnector(config)
        assert connector.config.bot_token == ""
        assert connector.config.chat_id == ""
        assert mock_warning.call_count >= 2


@pytest.fixture
def connector_with_credentials():
    """Create a connector with mocked credentials."""
    config = TelegramAPIConfig(bot_token="test-bot-token", chat_id="123456789")
    return TelegramAPIConnector(config)


@pytest.mark.asyncio
async def test_connect_without_credentials_returns_early():
    """Test that connect returns early without credentials."""
    config = TelegramAPIConfig(bot_token="", chat_id="")
    connector = TelegramAPIConnector(config)

    with patch("actions.telegram.connector.telegramAPI.logging.error") as mock_error:
        input_obj = TelegramInput(action="Test")
        await connector.connect(input_obj)
        mock_error.assert_called_with("Telegram credentials not configured")


@pytest.mark.asyncio
async def test_connect_logs_message(connector_with_credentials):
    """Test that connect logs the message being sent."""
    with patch(
        "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
    ) as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": {"message_id": 12345}})

        mock_post = AsyncMock(return_value=mock_response)
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance

        with patch("actions.telegram.connector.telegramAPI.logging.info") as mock_info:
            input_obj = TelegramInput(action="Test notification")
            await connector_with_credentials.connect(input_obj)
            mock_info.assert_any_call("SendThisToTelegram: Test notification")


@pytest.mark.asyncio
async def test_connect_uses_correct_api_url(connector_with_credentials):
    """Test that connect calls correct Telegram API URL."""
    with patch(
        "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
    ) as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": {"message_id": 12345}})

        mock_post = AsyncMock(return_value=mock_response)
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=mock_post)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)

        mock_session.return_value = mock_session_instance

        input_obj = TelegramInput(action="Test")
        await connector_with_credentials.connect(input_obj)

        mock_session_instance.post.assert_called_once()
        call_args = mock_session_instance.post.call_args
        assert "api.telegram.org" in call_args[0][0]
        assert "test-bot-token" in call_args[0][0]
