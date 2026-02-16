from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pydantic import ValidationError

from actions.telegram.connector.telegramAPI import (
    TelegramAPIConfig,
    TelegramAPIConnector,
)
from actions.telegram.interface import TelegramInput


class TestTelegramAPIConfig:
    """Test TelegramAPIConfig configuration."""

    def test_config_with_credentials(self):
        """Test configuration with bot token and chat ID."""
        config = TelegramAPIConfig(
            bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            chat_id="-1001234567890",
        )
        assert config.bot_token == "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        assert config.chat_id == "-1001234567890"

    def test_config_fields_required(self):
        """Test that bot_token and chat_id fields are required."""
        with pytest.raises(ValidationError):
            TelegramAPIConfig()  # type: ignore[call-arg]


class TestTelegramAPIConnectorInit:
    """Test TelegramAPIConnector initialization."""

    def test_init_with_valid_config(self):
        """Test initialization with valid credentials."""
        config = TelegramAPIConfig(bot_token="test_token", chat_id="test_chat")
        connector = TelegramAPIConnector(config)
        assert connector.config.bot_token == "test_token"
        assert connector.config.chat_id == "test_chat"

    def test_init_logs_warning_missing_token(self):
        """Test initialization logs warning when bot token is missing."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            config = TelegramAPIConfig(bot_token="", chat_id="test_chat")
            TelegramAPIConnector(config)
            mock_logging.warning.assert_any_call(
                "Telegram Bot Token not provided in configuration"
            )

    def test_init_logs_warning_missing_chat_id(self):
        """Test initialization logs warning when chat ID is missing."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            config = TelegramAPIConfig(bot_token="test_token", chat_id="")
            TelegramAPIConnector(config)
            mock_logging.warning.assert_any_call(
                "Telegram Chat ID not provided in configuration"
            )

    def test_init_logs_both_warnings_when_both_missing(self):
        """Test initialization logs both warnings when both credentials missing."""
        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            config = TelegramAPIConfig(bot_token="", chat_id="")
            TelegramAPIConnector(config)
            assert mock_logging.warning.call_count == 2


class TestTelegramAPIConnectorConnect:
    """Test connect method for sending messages."""

    @pytest.fixture
    def valid_config(self):
        """Create valid configuration."""
        return TelegramAPIConfig(
            bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            chat_id="-1001234567890",
        )

    @pytest.fixture
    def invalid_config(self):
        """Create configuration with missing credentials."""
        return TelegramAPIConfig(bot_token="", chat_id="")

    @pytest.mark.asyncio
    async def test_connect_missing_credentials_logs_error(self, invalid_config):
        """Test connect with missing credentials logs error and returns early."""
        connector = TelegramAPIConnector(invalid_config)
        telegram_input = TelegramInput(action="Test message")

        with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
            await connector.connect(telegram_input)
            mock_logging.error.assert_called_once_with(
                "Telegram credentials not configured"
            )

    @pytest.mark.asyncio
    async def test_connect_successful_message_send(self, valid_config):
        """Test successful message send returns message ID."""
        connector = TelegramAPIConnector(valid_config)
        telegram_input = TelegramInput(action="Test message")

        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 12345}}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_client:
            mock_client.return_value = mock_session

            with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
                await connector.connect(telegram_input)

                # Verify logging
                mock_logging.info.assert_any_call("SendThisToTelegram: Test message")
                mock_logging.info.assert_any_call(
                    "Telegram message sent successfully! Message ID: 12345"
                )

                # Verify API call
                call_args = mock_session.post.call_args
                expected_url = "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/sendMessage"
                assert call_args[0][0] == expected_url
                assert call_args[1]["json"]["chat_id"] == "-1001234567890"
                assert call_args[1]["json"]["text"] == "Test message"
                assert call_args[1]["json"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_connect_api_error_response(self, valid_config):
        """Test handling of Telegram API error response."""
        connector = TelegramAPIConnector(valid_config)
        telegram_input = TelegramInput(action="Test message")

        # Create mock response for error
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(
            return_value='{"ok":false,"error_code":400,"description":"Bad Request"}'
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_client:
            mock_client.return_value = mock_session

            with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
                await connector.connect(telegram_input)

                # Verify error logging
                mock_logging.error.assert_called()
                error_msg = mock_logging.error.call_args[0][0]
                assert "Telegram API error: 400" in error_msg

    @pytest.mark.asyncio
    async def test_connect_network_exception(self, valid_config):
        """Test handling of network exceptions."""
        connector = TelegramAPIConnector(valid_config)
        telegram_input = TelegramInput(action="Test message")

        # Create mock session that raises exception
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_client:
            mock_client.return_value = mock_session

            with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
                with pytest.raises(aiohttp.ClientError):
                    await connector.connect(telegram_input)

                # Verify error logging
                mock_logging.error.assert_called()
                error_msg = mock_logging.error.call_args[0][0]
                assert "Failed to send Telegram message" in error_msg

    @pytest.mark.asyncio
    async def test_connect_with_html_formatting(self, valid_config):
        """Test message with HTML formatting is sent correctly."""
        connector = TelegramAPIConnector(valid_config)
        telegram_input = TelegramInput(action="<b>Bold</b> text")

        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 12345}}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_client:
            mock_client.return_value = mock_session

            await connector.connect(telegram_input)

            # Verify HTML parse_mode is set
            call_args = mock_session.post.call_args
            assert call_args[1]["json"]["parse_mode"] == "HTML"
            assert call_args[1]["json"]["text"] == "<b>Bold</b> text"

    @pytest.mark.asyncio
    async def test_connect_logs_message_text(self, valid_config):
        """Test connect logs the message text being sent."""
        connector = TelegramAPIConnector(valid_config)
        telegram_input = TelegramInput(action="Robot status update")

        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "result": {"message_id": 99999}}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "actions.telegram.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_client:
            mock_client.return_value = mock_session

            with patch("actions.telegram.connector.telegramAPI.logging") as mock_logging:
                await connector.connect(telegram_input)

                # Verify message text is logged
                mock_logging.info.assert_any_call(
                    "SendThisToTelegram: Robot status update"
                )
