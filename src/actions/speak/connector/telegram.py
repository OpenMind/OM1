"""
Telegram Speak Connector - Send agent responses to Telegram.

This connector allows OM1 agents to send responses back to Telegram users.
It works in conjunction with the TelegramInput plugin.
"""

import asyncio
import logging
import os
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput

# Try to import telegram library
try:
    from telegram import Bot

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramSpeakConfig(ActionConfig):
    """
    Configuration for Telegram Speak Connector.

    Parameters
    ----------
    bot_token : Optional[str]
        Telegram Bot Token. Falls back to TELEGRAM_BOT_TOKEN env var.
    default_chat_id : Optional[int]
        Default chat ID to send messages to.
    """

    bot_token: Optional[str] = Field(
        default=None,
        description="Telegram Bot Token. Falls back to TELEGRAM_BOT_TOKEN env var.",
    )
    default_chat_id: Optional[int] = Field(
        default=None,
        description="Default Telegram chat ID to send messages to.",
    )


class TelegramSpeakConnector(ActionConnector[TelegramSpeakConfig, SpeakInput]):
    """
    A "Speak" connector that sends messages to Telegram.

    This connector sends the agent's spoken responses to Telegram users.
    """

    def __init__(self, config: TelegramSpeakConfig):
        """
        Initialize the Telegram speak connector.

        Parameters
        ----------
        config : TelegramSpeakConfig
            Configuration for the connector.
        """
        super().__init__(config)

        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. "
                "Install with: pip install python-telegram-bot"
            )

        # Get bot token
        self.bot_token = config.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError(
                "Telegram bot token not provided. "
                "Set TELEGRAM_BOT_TOKEN in .env or provide bot_token in config."
            )

        self.default_chat_id = config.default_chat_id or os.getenv(
            "TELEGRAM_CHAT_ID", None
        )
        if self.default_chat_id:
            self.default_chat_id = int(self.default_chat_id)

        self.bot = Bot(token=self.bot_token)
        self._active_chat_id: Optional[int] = None

        logging.info("Telegram Speak Connector initialized")

    def set_active_chat(self, chat_id: int):
        """Set the active chat ID for responses."""
        self._active_chat_id = chat_id

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Send a speak action to Telegram.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to send.
        """
        text = output_interface.action

        if not text or text.strip() == "":
            logging.debug("Empty speak action, skipping Telegram message")
            return

        # Determine which chat to send to
        chat_id = self._active_chat_id or self.default_chat_id

        if not chat_id:
            logging.warning(
                "No Telegram chat ID available. "
                "Set TELEGRAM_CHAT_ID or wait for a user to message the bot."
            )
            return

        try:
            await self.bot.send_message(chat_id=chat_id, text=text)
            logging.info(f"Sent to Telegram ({chat_id}): {text[:50]}...")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
