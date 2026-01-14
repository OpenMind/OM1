"""Tests for Discord webhook action."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.discord.connector.webhook import (
    DiscordWebhookConfig,
    DiscordWebhookConnector,
)
from actions.discord.interface import Discord, DiscordInput


class TestDiscordInput:
    """Tests for DiscordInput dataclass."""

    def test_default_values(self):
        """Test DiscordInput with default values."""
        input_obj = DiscordInput()
        assert input_obj.action == ""

    def test_with_value(self):
        """Test DiscordInput with custom value."""
        input_obj = DiscordInput(action="Hello from robot!")
        assert input_obj.action == "Hello from robot!"

    def test_with_markdown(self):
        """Test DiscordInput with Discord markdown."""
        input_obj = DiscordInput(action="**Bold** and *italic* text")
        assert "**Bold**" in input_obj.action
        assert "*italic*" in input_obj.action


class TestDiscordInterface:
    """Tests for Discord interface."""

    def test_interface_creation(self):
        """Test Discord interface creation."""
        input_obj = DiscordInput(action="Test message")
        output_obj = DiscordInput(action="Test message")
        message = Discord(input=input_obj, output=output_obj)
        assert message.input.action == "Test message"
        assert message.output.action == "Test message"


class TestDiscordWebhookConfig:
    """Tests for DiscordWebhookConfig."""

    def test_with_webhook_url(self):
        """Test config with webhook URL."""
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        assert config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        assert config.username is None
        assert config.avatar_url is None

    def test_with_all_options(self):
        """Test config with all options."""
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="RobotBot",
            avatar_url="https://example.com/avatar.png",
        )
        assert config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        assert config.username == "RobotBot"
        assert config.avatar_url == "https://example.com/avatar.png"


class TestDiscordWebhookConnector:
    """Tests for DiscordWebhookConnector."""

    def test_init_with_webhook_url(self):
        """Test initialization with webhook URL."""
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        connector = DiscordWebhookConnector(config)
        assert (
            connector.config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        )

    def test_init_without_webhook_url(self):
        """Test initialization without webhook URL logs warning."""
        with patch("actions.discord.connector.webhook.logging.warning") as mock_warning:
            config = DiscordWebhookConfig(webhook_url="")
            DiscordWebhookConnector(config)
            mock_warning.assert_called_with(
                "Discord webhook URL not provided in configuration"
            )


class TestDiscordWebhookConnectorConnect:
    """Tests for DiscordWebhookConnector.connect method."""

    @pytest.fixture
    def connector_with_url(self):
        """Create a connector with webhook URL."""
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        return DiscordWebhookConnector(config)

    @pytest.fixture
    def connector_with_options(self):
        """Create a connector with all options."""
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="TestBot",
            avatar_url="https://example.com/avatar.png",
        )
        return DiscordWebhookConnector(config)

    @pytest.mark.asyncio
    async def test_connect_without_webhook_url(self):
        """Test that connect returns early without webhook URL."""
        config = DiscordWebhookConfig(webhook_url="")
        connector = DiscordWebhookConnector(config)

        with patch("actions.discord.connector.webhook.logging.error") as mock_error:
            input_obj = DiscordInput(action="Test")
            await connector.connect(input_obj)
            mock_error.assert_called_with("Discord webhook URL not configured")

    @pytest.mark.asyncio
    async def test_connect_with_empty_message(self, connector_with_url):
        """Test that connect skips empty message."""
        with patch("actions.discord.connector.webhook.logging.warning") as mock_warning:
            input_obj = DiscordInput(action="")
            await connector_with_url.connect(input_obj)
            mock_warning.assert_called_with("Empty Discord message, skipping send")

    @pytest.mark.asyncio
    async def test_connect_logs_message(self, connector_with_url):
        """Test that connect logs the message being sent."""
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch("actions.discord.connector.webhook.logging.info") as mock_info:
                input_obj = DiscordInput(action="Test notification")
                await connector_with_url.connect(input_obj)
                mock_info.assert_any_call("SendThisToDiscord: Test notification")

    @pytest.mark.asyncio
    async def test_connect_sends_correct_payload(self, connector_with_url):
        """Test that connect sends correct JSON payload."""
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            input_obj = DiscordInput(action="Hello Discord!")
            await connector_with_url.connect(input_obj)

            mock_session_instance.post.assert_called_once()
            call_args = mock_session_instance.post.call_args
            assert call_args[0][0] == "https://discord.com/api/webhooks/123/abc"
            assert call_args[1]["json"] == {"content": "Hello Discord!"}

    @pytest.mark.asyncio
    async def test_connect_includes_username_and_avatar(self, connector_with_options):
        """Test that connect includes username and avatar when configured."""
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            input_obj = DiscordInput(action="Hello!")
            await connector_with_options.connect(input_obj)

            call_args = mock_session_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["content"] == "Hello!"
            assert payload["username"] == "TestBot"
            assert payload["avatar_url"] == "https://example.com/avatar.png"

    @pytest.mark.asyncio
    async def test_connect_logs_success_on_204(self, connector_with_url):
        """Test that connect logs success on 204 response."""
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 204

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch("actions.discord.connector.webhook.logging.info") as mock_info:
                input_obj = DiscordInput(action="Test")
                await connector_with_url.connect(input_obj)

                success_logged = any(
                    "successfully" in str(call).lower()
                    for call in mock_info.call_args_list
                )
                assert success_logged

    @pytest.mark.asyncio
    async def test_connect_handles_error_response(self, connector_with_url):
        """Test that connect handles error responses."""
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad Request")

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch("actions.discord.connector.webhook.logging.error") as mock_error:
                input_obj = DiscordInput(action="Test")
                await connector_with_url.connect(input_obj)

                error_logged = any(
                    "400" in str(call) for call in mock_error.call_args_list
                )
                assert error_logged
