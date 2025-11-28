import asyncio
import logging
from typing import Any

import aiohttp

from actions.base import ActionConfig, ActionConnector
from actions.remember_location.interface import RememberLocationInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class UnitreeGo2RememberLocationConnector(ActionConnector[RememberLocationInput]):
    """
    Connector that persists a remembered location for Unitree Go2 by POSTing to an HTTP API.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the RememberLocationGo2Connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.base_url = getattr(
            config, "base_url", "http://localhost:5000/maps/locations/add/slam"
        )
        self.timeout = getattr(config, "timeout", 5)
        self.map_name = getattr(config, "map_name", "map")

        self.elevenlabs_provider = ElevenLabsTTSProvider()

    async def connect(self, input_protocol: RememberLocationInput) -> None:
        """
        Connect the input protocol to the remember location action for Go2.

        Parameters
        ----------
        input_protocol : RememberLocationInput
            The input protocol containing the action details.
        """
        if not self.base_url:
            logging.error("RememberLocationGo2 connector missing 'base_url' in config")
            return

        payload: dict[str, Any] = {
            "map_name": self.map_name,
            "label": input_protocol.action,
            "description": getattr(input_protocol, "description", ""),
        }

        headers = {"Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, json=payload, headers=headers, timeout=self.timeout
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 200 and resp.status < 300:
                        logging.info(
                            f"RememberLocationGo2: stored '{input_protocol.action}' -> {resp.status} {text}"
                        )
                        self.elevenlabs_provider.add_pending_message(
                            f"Location {input_protocol.action} remembered for Go2. Woof! Woof!"
                        )
                    else:
                        logging.error(
                            f"RememberLocationGo2 API returned {resp.status}: {text}"
                        )
        except asyncio.TimeoutError:
            logging.error("RememberLocationGo2 API request timed out")
        except Exception as e:
            logging.error(f"RememberLocationGo2 API request failed: {e}")
