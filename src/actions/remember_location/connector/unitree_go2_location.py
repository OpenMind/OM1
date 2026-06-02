import asyncio
import logging
from typing import Any, Optional

import aiohttp
from pydantic import Field, model_validator

from actions.base import ActionConfig, ActionConnector
from actions.remember_location.interface import RememberLocationInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class UnitreeGo2RememberLocationConfig(ActionConfig):
    """
    Configuration for Unitree Go2 Remember Location connector.

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
    def set_base_url(self) -> "UnitreeGo2RememberLocationConfig":
        """
        Set base_url based on use_sim if not explicitly provided.

        Returns
        -------
        UnitreeGo2RememberLocationConfig
            The validated configuration with base_url set if it was None.
        """
        if self.base_url is None:
            if self.use_sim:
                self.base_url = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/add/slam"
            else:
                self.base_url = "http://localhost:5000/maps/locations/add/slam"

        return self


class UnitreeGo2RememberLocationConnector(ActionConnector[UnitreeGo2RememberLocationConfig, RememberLocationInput]):
    """
    Connector that persists a remembered location for Unitree Go2 by POSTing to an HTTP API.
    """

    def __init__(self, config: UnitreeGo2RememberLocationConfig):
        """
        Initialize the RememberLocationGo2Connector.

        Parameters
        ----------
        config : UnitreeGo2RememberLocationConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.timeout = self.config.timeout
        self.map_name = self.config.map_name

        self.elevenlabs_provider = ElevenLabsTTSProvider()

    async def connect(self, output_interface: RememberLocationInput) -> None:
        """
        Connect the input protocol to the remember location action for Go2.

        Parameters
        ----------
        output_interface : RememberLocationInput
            The input protocol containing the action details.
        """
        if not self.base_url:
            logging.error("RememberLocationGo2 connector missing 'base_url' in config")
            return

        payload: dict[str, Any] = {
            "map_name": self.map_name,
            "label": output_interface.action,
            "description": getattr(output_interface, "description", ""),
        }

        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

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
                        logging.info(f"RememberLocationGo2: stored '{output_interface.action}' -> {resp.status} {text}")
                        self.elevenlabs_provider.add_pending_message(
                            f"Location {output_interface.action} remembered for Go2. Woof! Woof!"
                        )
                    else:
                        logging.error(f"RememberLocationGo2 API returned {resp.status}: {text}")
        except asyncio.TimeoutError:
            logging.error("RememberLocationGo2 API request timed out")
        except Exception as e:
            logging.error(f"RememberLocationGo2 API request failed: {e}")
