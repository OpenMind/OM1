import logging
from typing import Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.discord.interface import DiscordInput

DISCORD_MAX_CONTENT_LENGTH = 2000


class DiscordWebhookConfig(ActionConfig):
    """
    Configuration class for Discord Webhook Connector.

    Parameters
    ----------
    webhook_url : str
        Discord webhook URL for sending messages.
    username : str, optional
        Override the default webhook username.
    avatar_url : str, optional
        Override the default webhook avatar URL.
    """

    webhook_url: str = Field(description="Discord webhook URL")
    username: Optional[str] = Field(
        default=None, description="Override webhook username"
    )
    avatar_url: Optional[str] = Field(
        default=None, description="Override webhook avatar URL"
    )


class DiscordWebhookConnector(ActionConnector[DiscordWebhookConfig, DiscordInput]):
    """
    Connector for Discord Webhook API.

    This connector sends messages to Discord channels using webhooks.
    Webhooks provide a simple way to post messages without a bot token.
    """

    def __init__(self, config: DiscordWebhookConfig):
        """
        Initialize the Discord Webhook connector.

        Parameters
        ----------
        config : DiscordWebhookConfig
            Configuration object for the connector.
        """
        super().__init__(config)

        if not self.config.webhook_url:
            logging.warning("Discord webhook URL not provided in configuration")

    async def connect(self, output_interface: DiscordInput) -> None:
        """
        Send message via Discord Webhook.

        Parameters
        ----------
        output_interface : DiscordInput
            The DiscordInput interface containing the message text.
        """
        if not self.config.webhook_url:
            logging.error("Discord webhook URL not configured")
            return

        message_text = output_interface.action
        if not message_text:
            logging.warning("Empty Discord message, skipping send")
            return

        if len(message_text) > DISCORD_MAX_CONTENT_LENGTH:
            logging.warning(
                f"Discord message truncated from {len(message_text)} "
                f"to {DISCORD_MAX_CONTENT_LENGTH} characters"
            )
            message_text = message_text[:DISCORD_MAX_CONTENT_LENGTH]

        try:
            logging.info(f"SendThisToDiscord: {message_text}")

            payload: dict = {"content": message_text}

            if self.config.username:
                payload["username"] = self.config.username
            if self.config.avatar_url:
                payload["avatar_url"] = self.config.avatar_url

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                ) as response:
                    if response.status in (200, 204):
                        logging.info("Discord message sent successfully!")
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"Discord webhook error: {response.status} - {error_text}"
                        )
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=error_text,
                        )

        except aiohttp.ClientResponseError:
            raise
        except aiohttp.ClientError as e:
            logging.error(f"Network error sending Discord message: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Failed to send Discord message: {str(e)}")
            raise
