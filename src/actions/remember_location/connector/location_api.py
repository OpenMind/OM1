import asyncio
import logging
from typing import Any

import aiohttp

from actions.base import ActionConfig, ActionConnector
from actions.remember_location.interface import RememberLocationInput


class RememberLocationConnector(ActionConnector[RememberLocationInput]):
    """
    Connector that persists a remembered location by POSTing to an HTTP API.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the RememberLocationConnector.

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

    async def connect(self, input_protocol: RememberLocationInput) -> None:
        """
        Connect the input protocol to the remember location action.

        Parameters
        ----------
        input_protocol : RememberLocationInput
            The input protocol containing the action details.
        """
        if not self.base_url:
            logging.error("RememberLocation connector missing 'base_url' in config")
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
                            f"RememberLocation: stored '{input_protocol.action}' -> {resp.status} {text}"
                        )
                    else:
                        logging.error(
                            f"RememberLocation API returned {resp.status}: {text}"
                        )
        except asyncio.TimeoutError:
            logging.error("RememberLocation API request timed out")
        except Exception as e:
            logging.error(f"RememberLocation API request failed: {e}")
