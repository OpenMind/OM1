import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import requests
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class HomeAssistantConfig(SensorConfig):
    """
    Configuration for Home Assistant integration.

    Parameters
    ----------
    ha_url : str
        Home Assistant URL (e.g., http://homeassistant.local:8123)
    ha_token : str
        Home Assistant long-lived access token
    poll_interval : float
        Seconds between polling for new commands
    """

    ha_url: str = Field(default="", description="Home Assistant URL")
    ha_token: str = Field(default="", description="Home Assistant access token")
    poll_interval: float = Field(default=2.0, description="Polling interval in seconds")


class HomeAssistant(FuserInput[HomeAssistantConfig, Dict]):
    """
    Home Assistant integration for voice commands and orders.

    Monitors Home Assistant for incoming voice commands, extracts order
    information, and reports them to the OM1 agent.
    """

    def __init__(self, config: HomeAssistantConfig):
        super().__init__(config)

        # Get config from environment or config
        self.ha_url = config.ha_url or os.environ.get("HOME_ASSISTANT_URL", "")
        self.ha_token = config.ha_token or os.environ.get("HOME_ASSISTANT_TOKEN", "")
        self.poll_interval = config.poll_interval

        if not self.ha_url or not self.ha_token:
            logging.error(
                "HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN must be configured"
            )

        # Track IO
        self.io_provider = IOProvider()
        self.messages: List[Message] = []

        # Track last command to avoid duplicates
        self.last_command_id = None

        logging.info("HomeAssistant: Initialized")

    async def _poll(self) -> Dict:
        """
        Poll Home Assistant for new voice commands.

        Returns
        -------
        Dict
            Command data with keys: command, timestamp, command_id
        """
        await asyncio.sleep(self.poll_interval)

        try:
            # Query Home Assistant REST API for latest voice command
            # This assumes you have a sensor/input_text entity that stores commands
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }

            # Get state of command entity (adjust entity_id as needed)
            response = requests.get(
                f"{self.ha_url}/api/states/input_text.om1_voice_command",
                headers=headers,
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                command = data.get("state", "")
                last_changed = data.get("last_changed")

                # Only return new commands
                if command and last_changed != self.last_command_id:
                    self.last_command_id = last_changed
                    return {
                        "command": command,
                        "timestamp": time.time(),
                        "command_id": last_changed,
                    }

            return {}

        except Exception as e:
            logging.error(f"Error polling Home Assistant: {e}")
            return {}

    async def _raw_to_text(self, raw_input: Dict) -> Optional[Message]:
        """
        Convert voice command to Message.

        Parameters
        ----------
        raw_input : Dict
            Command data from Home Assistant

        Returns
        -------
        Message
            Formatted voice command message
        """
        if not raw_input or "command" not in raw_input:
            return None

        command = raw_input["command"]
        timestamp = raw_input["timestamp"]

        message = f"Voice command from Home Assistant: {command}"
        logging.info(f"HomeAssistant: {message}")

        return Message(timestamp=timestamp, message=message)

    async def raw_to_text(self, raw_input: Dict):
        """
        Process voice command and add to message buffer.

        Parameters
        ----------
        raw_input : Dict
            Raw command data
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the message buffer.

        Returns
        -------
        Optional[str]
            Formatted commands or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        # Combine all commands
        commands = [msg.message for msg in self.messages]
        last_timestamp = self.messages[-1].timestamp

        result_message = "\n".join(commands)

        result = f"""
{self.__class__.__name__} INPUT
// START
{result_message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, result_message, last_timestamp
        )
        self.messages = []
        return result
