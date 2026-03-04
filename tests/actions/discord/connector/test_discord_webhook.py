"""Tests for discord webhook action."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from actions.discord.connector.webhook import (
    DISCORD_MAX_CONTENT_LENGTH,
    DiscordWebhookConfig,
    DiscordWebhookConnector,
)
from actions.discord.interface import Discord, DiscordInput


class TestDiscordInput:
    def test_default_values(self):
        input_obj = DiscordInput()
        assert input_obj.action == ""

    def test_with_value(self):
        input_obj = DiscordInput(action="Hello from robot!")
        assert input_obj.action == "Hello from robot!"

    def test_with_markdown(self):
        input_obj = DiscordInput(action="**Bold** and *italic* text")
        assert "**Bold**" in input_obj.action
        assert "*italic*" in input_obj.action


class TestDiscordInterface:
    def test_interface_creation(self):
        input_obj = DiscordInput(action="Test message")
        output_obj = DiscordInput(action="Test message")
        message = Discord(input=input_obj, output=output_obj)
        assert message.input.action == "Test message"
        assert message.output.action == "Test message"


class TestDiscordWebhookConfig:
    def test_with_webhook_url(self):
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        assert config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        assert config.username is None
        assert config.avatar_url is None

    def test_with_all_options(self):
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="RobotBot",
            avatar_url="https://example.com/avatar.png",
        )
        assert config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        assert config.username == "RobotBot"
        assert config.avatar_url == "https://example.com/avatar.png"


class TestDiscordWebhookConnector:
    def test_init_with_webhook_url(self):
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        connector = DiscordWebhookConnector(config)
        assert (
            connector.config.webhook_url == "https://discord.com/api/webhooks/123/abc"
        )

    def test_init_without_webhook_url(self):
        with patch("actions.discord.connector.webhook.logging.warning") as mock_warning:
            config = DiscordWebhookConfig(webhook_url="")
            DiscordWebhookConnector(config)
            mock_warning.assert_called_with(
                "Discord webhook URL not provided in configuration"
            )


class TestDiscordWebhookConnectorConnect:
    @pytest.fixture
    def connector_with_url(self):
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc"
        )
        return DiscordWebhookConnector(config)

    @pytest.fixture
    def connector_with_options(self):
        config = DiscordWebhookConfig(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="TestBot",
            avatar_url="https://example.com/avatar.png",
        )
        return DiscordWebhookConnector(config)

    @pytest.fixture
    def mock_discord_session(self):
        """Reusable aiohttp.ClientSession mock for Discord webhook tests."""
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

            yield {
                "session": mock_session,
                "session_instance": mock_session_instance,
                "response": mock_response,
            }

    @pytest.mark.asyncio
    async def test_connect_without_webhook_url(self):
        config = DiscordWebhookConfig(webhook_url="")
        connector = DiscordWebhookConnector(config)

        with patch("actions.discord.connector.webhook.logging.error") as mock_error:
            input_obj = DiscordInput(action="Test")
            await connector.connect(input_obj)
            mock_error.assert_called_with("Discord webhook URL not configured")

    @pytest.mark.asyncio
    async def test_connect_with_empty_message(self, connector_with_url):
        with patch("actions.discord.connector.webhook.logging.warning") as mock_warning:
            input_obj = DiscordInput(action="")
            await connector_with_url.connect(input_obj)
            mock_warning.assert_called_with("Empty Discord message, skipping send")

    @pytest.mark.asyncio
    async def test_connect_logs_message(self, connector_with_url, mock_discord_session):
        with patch("actions.discord.connector.webhook.logging.info") as mock_info:
            input_obj = DiscordInput(action="Test notification")
            await connector_with_url.connect(input_obj)
            mock_info.assert_any_call("SendThisToDiscord: Test notification")

    @pytest.mark.asyncio
    async def test_connect_sends_correct_payload(
        self, connector_with_url, mock_discord_session
    ):
        input_obj = DiscordInput(action="Hello Discord!")
        await connector_with_url.connect(input_obj)

        session_instance = mock_discord_session["session_instance"]
        session_instance.post.assert_called_once()
        call_args = session_instance.post.call_args
        assert call_args[0][0] == "https://discord.com/api/webhooks/123/abc"
        assert call_args[1]["json"] == {"content": "Hello Discord!"}

    @pytest.mark.asyncio
    async def test_connect_includes_username_and_avatar(
        self, connector_with_options, mock_discord_session
    ):
        input_obj = DiscordInput(action="Hello!")
        await connector_with_options.connect(input_obj)

        session_instance = mock_discord_session["session_instance"]
        call_args = session_instance.post.call_args
        payload = call_args[1]["json"]
        assert payload["content"] == "Hello!"
        assert payload["username"] == "TestBot"
        assert payload["avatar_url"] == "https://example.com/avatar.png"

    @pytest.mark.asyncio
    async def test_connect_logs_success_on_204(
        self, connector_with_url, mock_discord_session
    ):
        mock_discord_session["response"].status = 204

        with patch("actions.discord.connector.webhook.logging.info") as mock_info:
            input_obj = DiscordInput(action="Test")
            await connector_with_url.connect(input_obj)

            success_logged = any(
                "successfully" in str(call).lower() for call in mock_info.call_args_list
            )
            assert success_logged

    @pytest.mark.asyncio
    async def test_connect_logs_success_on_200(
        self, connector_with_url, mock_discord_session
    ):
        mock_discord_session["response"].status = 200

        with patch("actions.discord.connector.webhook.logging.info") as mock_info:
            input_obj = DiscordInput(action="Test")
            await connector_with_url.connect(input_obj)

            success_logged = any(
                "successfully" in str(call).lower() for call in mock_info.call_args_list
            )
            assert success_logged

    @pytest.mark.asyncio
    async def test_connect_raises_on_error_response(
        self, connector_with_url, mock_discord_session
    ):
        mock_discord_session["response"].status = 400
        mock_discord_session["response"].text = AsyncMock(return_value="Bad Request")
        mock_discord_session["response"].request_info = MagicMock()
        mock_discord_session["response"].history = ()

        with pytest.raises(aiohttp.ClientResponseError):
            input_obj = DiscordInput(action="Test")
            await connector_with_url.connect(input_obj)

    @pytest.mark.asyncio
    async def test_connect_raises_on_rate_limit(
        self, connector_with_url, mock_discord_session
    ):
        mock_discord_session["response"].status = 429
        mock_discord_session["response"].text = AsyncMock(return_value="Rate limited")
        mock_discord_session["response"].request_info = MagicMock()
        mock_discord_session["response"].history = ()

        with pytest.raises(aiohttp.ClientResponseError):
            input_obj = DiscordInput(action="Test")
            await connector_with_url.connect(input_obj)

    @pytest.mark.asyncio
    async def test_connect_handles_network_error(self, connector_with_url):
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_session.side_effect = aiohttp.ClientError("Connection refused")

            with pytest.raises(aiohttp.ClientError):
                input_obj = DiscordInput(action="Test")
                await connector_with_url.connect(input_obj)

    @pytest.mark.asyncio
    async def test_connect_handles_generic_exception(self, connector_with_url):
        with patch(
            "actions.discord.connector.webhook.aiohttp.ClientSession"
        ) as mock_session:
            mock_session.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(RuntimeError):
                input_obj = DiscordInput(action="Test")
                await connector_with_url.connect(input_obj)

    @pytest.mark.asyncio
    async def test_connect_truncates_long_message(
        self, connector_with_url, mock_discord_session
    ):
        long_message = "x" * 2500
        input_obj = DiscordInput(action=long_message)
        await connector_with_url.connect(input_obj)

        session_instance = mock_discord_session["session_instance"]
        call_args = session_instance.post.call_args
        sent_content = call_args[1]["json"]["content"]
        assert len(sent_content) == DISCORD_MAX_CONTENT_LENGTH

    @pytest.mark.asyncio
    async def test_connect_does_not_truncate_short_message(
        self, connector_with_url, mock_discord_session
    ):
        short_message = "x" * 100
        input_obj = DiscordInput(action=short_message)
        await connector_with_url.connect(input_obj)

        session_instance = mock_discord_session["session_instance"]
        call_args = session_instance.post.call_args
        sent_content = call_args[1]["json"]["content"]
        assert len(sent_content) == 100
