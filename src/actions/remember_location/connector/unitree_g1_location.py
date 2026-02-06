import asyncio
import logging
from typing import Any

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.remember_location.interface import RememberLocationInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class UnitreeG1RememberLocationConfig(ActionConfig):
    """
    Configuration for Unitree G1 Remember Location connector.

    Parameters
    ----------
    base_url : str
        The base URL for the remember location API.
    timeout : int
        Timeout for the HTTP requests in seconds.
    map_name : str
        The name of the map to use when remembering locations.
    """

    base_url: str = Field(
        default="http://localhost:5000/maps/locations/add/slam",
        description="The base URL for the remember location API.",
    )
    timeout: int = Field(
        default=5,
        description="Timeout for the HTTP requests in seconds.",
    )
    map_name: str = Field(
        default="map",
        description="The name of the map to use when remembering locations.",
    )


class UnitreeG1RememberLocationConnector(
    ActionConnector[UnitreeG1RememberLocationConfig, RememberLocationInput]
):
    """
    Connector that persists a remembered location for Unitree G1 by POSTing to an HTTP API.
    """

    def __init__(self, config: UnitreeG1RememberLocationConfig):
        """
        Initialize the RememberLocationG1Connector.

        Parameters
        ----------
        config : UnitreeG1RememberLocationConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.base_url = self.config.base_url
        self.timeout = self.config.timeout
        self.map_name = self.config.map_name

        self.elevenlabs_provider = ElevenLabsTTSProvider()

    async def connect(self, output_interface: RememberLocationInput) -> None:
        """
        Connect the input protocol to the remember location action for G1.

        Parameters
        ----------
        output_interface : RememberLocationInput
            The input protocol containing the action details.
        """
        if not self.base_url:
            logging.error("RememberLocationG1 connector missing 'base_url' in config")
            return

        payload: dict[str, Any] = {
            "map_name": self.map_name,
            "label": output_interface.action,
            "description": getattr(output_interface, "description", ""),
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
                            f"RememberLocationG1: stored '{output_interface.action}' -> {resp.status} {text}"
                        )
                        self.elevenlabs_provider.add_pending_message(
                            f"Location {output_interface.action} remembered !"
                        )
                    else:
                        logging.error(
                            f"RememberLocationG1 API returned {resp.status}: {text}"
                        )
        except asyncio.TimeoutError:
            logging.error("RememberLocationG1 API request timed out")
        except Exception as e:
            logging.error(f"RememberLocationG1 API request failed: {e}")
