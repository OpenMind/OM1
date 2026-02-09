import logging

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.telegram.interface import TelegramInput


class TelegramAPIConfig(ActionConfig):
    """
    Configuration class for TelegramAPIConnector.
    """

    bot_token: str = Field(description="Telegram Bot Token for authentication")
    chat_id: str = Field(description="Telegram Chat ID to send messages to")


class TelegramAPIConnector(ActionConnector[TelegramAPIConfig, TelegramInput]):
    """
    Connector for Telegram Bot API.

    This connector integrates with Telegram Bot API to send messages from the robot.
    """

    def __init__(self, config: TelegramAPIConfig):
        """
        Initialize the Telegram API connector.

        Parameters
        ----------
        config : TelegramAPIConfig
            Configuration object for the connector.
        """
        super().__init__(config)

        if not self.config.bot_token:
            logging.warning("Telegram Bot Token not provided in configuration")
        if not self.config.chat_id:
            logging.warning("Telegram Chat ID not provided in configuration")

    async def connect(self, output_interface: TelegramInput) -> None:
        """
        Send message via Telegram Bot API.

        Parameters
        ----------
        output_interface : TelegramInput
            The TelegramInput interface containing the message text.
        """
        if not self.config.bot_token or not self.config.chat_id:
            logging.error("Telegram credentials not configured")
            return

        try:
            message_text = output_interface.action
            logging.info(f"SendThisToTelegram: {message_text}")

            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.chat_id,
                "text": message_text,
                "parse_mode": "HTML",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        message_id = data.get("result", {}).get("message_id")
                        logging.info(
                            f"Telegram message sent successfully! Message ID: {message_id}"
                        )
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"Telegram API error: {response.status} - {error_text}"
                        )

        except Exception as e:
            logging.error(f"Failed to send Telegram message: {str(e)}")
            raise
