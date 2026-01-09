"""
Telegram Input Plugin - Receive messages from Telegram Bot.

This plugin allows OM1 agents to receive messages from Telegram users,
enabling remote interaction with your AI agent through Telegram.

Setup:
1. Create a bot via @BotFather on Telegram
2. Get your bot token
3. Add the token to your .env file as TELEGRAM_BOT_TOKEN
4. Add TelegramInput to your config's agent_inputs
"""

import asyncio
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

# Try to import telegram library
try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning(
        "python-telegram-bot not installed. "
        "Install with: pip install python-telegram-bot"
    )


class TelegramInputConfig(SensorConfig):
    """
    Configuration for Telegram Input.

    Parameters
    ----------
    bot_token : Optional[str]
        Telegram Bot Token from @BotFather.
        If not provided, reads from TELEGRAM_BOT_TOKEN env var.
    input_name : str
        Name of the input for LLM context.
    allowed_users : Optional[List[int]]
        List of allowed Telegram user IDs. If empty, allows all users.
    """

    bot_token: Optional[str] = Field(
        default=None,
        description="Telegram Bot Token. Falls back to TELEGRAM_BOT_TOKEN env var.",
    )
    input_name: str = Field(
        default="Telegram Message",
        description="Name of the input for LLM context",
    )
    allowed_users: Optional[List[int]] = Field(
        default=None,
        description="List of allowed Telegram user IDs. None = allow all.",
    )


class TelegramInput(FuserInput[TelegramInputConfig, Optional[str]]):
    """
    Telegram Input - Receive messages from Telegram Bot.

    This plugin connects to Telegram Bot API and receives messages
    from users, passing them to the LLM for processing.
    """

    def __init__(self, config: TelegramInputConfig):
        """Initialize TelegramInput instance."""
        super().__init__(config)

        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. "
                "Install with: pip install python-telegram-bot"
            )

        # Get bot token from config or environment
        self.bot_token = self.config.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError(
                "Telegram bot token not provided. "
                "Set TELEGRAM_BOT_TOKEN in .env or provide bot_token in config."
            )

        # Buffer for storing messages
        self.messages: List[Message] = []
        self.message_buffer: Queue[dict] = Queue()

        # IO Provider for tracking
        self.descriptor_for_LLM = self.config.input_name
        self.io_provider = IOProvider()

        # Allowed users (None = allow all)
        self.allowed_users = self.config.allowed_users

        # Telegram application
        self.application: Optional[Application] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.running = False

        # Store chat IDs for responses
        self.active_chats: dict = {}

        logging.info("Telegram Input initialized")

        # Start the bot in a separate thread
        self._start_bot_thread()

    def _start_bot_thread(self):
        """Start the Telegram bot in a separate thread."""
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()

    def _run_bot(self):
        """Run the Telegram bot (in separate thread)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._start_bot())
        except Exception as e:
            logging.error(f"Telegram bot error: {e}")
        finally:
            loop.close()

    async def _start_bot(self):
        """Start the Telegram bot application."""
        self.application = Application.builder().token(self.bot_token).build()

        # Add handlers
        self.application.add_handler(
            CommandHandler("start", self._handle_start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self._handle_help_command)
        )
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        self.running = True
        logging.info("Telegram bot started. Send /start to your bot to begin.")

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)

        # Keep running
        while self.running:
            await asyncio.sleep(1)

        # Cleanup
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to interact with the bot."""
        if self.allowed_users is None or len(self.allowed_users) == 0:
            return True
        return user_id in self.allowed_users

    async def _handle_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /start command."""
        if update.effective_user is None or update.effective_chat is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if not self._is_user_allowed(user_id):
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        # Store chat for responses
        self.active_chats[user_id] = chat_id

        username = update.effective_user.first_name or "there"
        await update.message.reply_text(
            f"Hello {username}! I'm connected to an OM1 AI agent.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help for more information."
        )
        logging.info(f"Telegram user {user_id} started chat")

    async def _handle_help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /help command."""
        if update.effective_user is None:
            return

        if not self._is_user_allowed(update.effective_user.id):
            return

        await update.message.reply_text(
            "OM1 Telegram Bot\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n\n"
            "Just send any message and I'll respond!"
        )

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle incoming text messages."""
        if update.effective_user is None or update.message is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if not self._is_user_allowed(user_id):
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        # Store chat for responses
        self.active_chats[user_id] = chat_id

        text = update.message.text
        username = update.effective_user.first_name or "User"

        logging.info(f"Telegram message from {username} ({user_id}): {text}")

        # Add message to buffer for LLM processing
        self.message_buffer.put(
            {
                "text": text,
                "user_id": user_id,
                "chat_id": chat_id,
                "username": username,
                "timestamp": time.time(),
            }
        )

        # Send typing indicator
        await update.message.chat.send_action("typing")

    async def send_response(self, chat_id: int, text: str):
        """Send a response to a Telegram chat."""
        if self.application and self.application.bot:
            try:
                await self.application.bot.send_message(chat_id=chat_id, text=text)
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

    async def _poll(self) -> Optional[str]:
        """Poll for new messages from Telegram."""
        await asyncio.sleep(0.5)

        try:
            msg_data = self.message_buffer.get_nowait()
            # Format: "Username says: message"
            return f"{msg_data['username']} says: {msg_data['text']}"
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Process raw input to generate a timestamped message."""
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw input to text and update message buffer."""
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    def get_latest_chat_id(self) -> Optional[int]:
        """Get the most recent chat ID for sending responses."""
        if self.active_chats:
            return list(self.active_chats.values())[-1]
        return None
