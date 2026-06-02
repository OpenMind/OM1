import asyncio
import logging
from typing import Any, Optional

import aiohttp
from pydantic import Field, model_validator

from actions.base import ActionConfig, ActionConnector
from actions.remember_location.interface import RememberLocationInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class UnitreeG1RememberLocationConfig(ActionConfig):
    """
    Configuration for Unitree G1 Remember Location connector.

    Parameters
    ----------
    base_url : Optional[str]
        Base URL for the remember location API. If None, automatically determined by use_sim flag.
    api_key : str
        API key for OpenMind API authentication.
    use_sim : bool
        Whether to run the connector in the simulator.
    timeout : int
        Timeout for the HTTP requests in seconds.
    map_name : str
        The name of the map to use when remembering locations.
    """

    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the remember location API. If None, determined by use_sim flag.",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind API authentication",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to run the connector in the simulator.",
    )
    timeout: int = Field(
        default=5,
        description="Timeout for the HTTP requests in seconds.",
    )
    map_name: str = Field(
        default="map",
        description="The name of the map to use when remembering locations.",
    )

    @model_validator(mode="after")
    def set_base_url(self) -> "UnitreeG1RememberLocationConfig":
        """
        Set base_url based on use_sim if not explicitly provided.

        Returns
        -------
        UnitreeG1RememberLocationConfig
            The validated configuration with base_url set if it was None.
        """
        if self.base_url is None:
            if self.use_sim:
                self.base_url = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/add/slam"
            else:
                self.base_url = "http://localhost:5000/maps/locations/add/slam"

        return self


class UnitreeG1RememberLocationConnector(ActionConnector[UnitreeG1RememberLocationConfig, RememberLocationInput]):
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
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 200 and resp.status < 300:
                        logging.info(f"RememberLocationG1: stored '{output_interface.action}' -> {resp.status} {text}")
                        self.elevenlabs_provider.add_pending_message(f"Location {output_interface.action} remembered !")
                    else:
                        logging.error(f"RememberLocationG1 API returned {resp.status}: {text}")
        except asyncio.TimeoutError:
            logging.error("RememberLocationG1 API request timed out")
        except Exception as e:
            logging.error(f"RememberLocationG1 API request failed: {e}")
