import logging
from typing import Any, Dict, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class StartNav2HookContext(BaseModel):
    """
    Context for starting Nav2 hook.

    Parameters
    ----------
    base_url : Optional[str]
        Base URL for the Nav2 system. If None, determined by use_sim flag.
    use_sim : bool
        Whether to run the connector in the simulator.
    map_name : str
        Name of the map to use for navigation.
    """

    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the SLAM system. If None, determined by use_sim flag.",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to run the connector in the simulator.",
    )
    map_name: str = Field(
        default="map",
        description="Name of the map to use for navigation",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind cloud system",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def set_base_url(self) -> "StartNav2HookContext":
        """
        Set base_url based on use_sim if not explicitly provided.

        Returns
        -------
        StartNav2HookContext
            The validated context with base_url set if it was None.
        """
        if self.base_url is None:
            if self.use_sim:
                self.base_url = "https://api.openmind.com/api/core/simulation/orchestrator"
            else:
                self.base_url = "http://localhost:5000"

        return self


class StopNav2HookContext(BaseModel):
    """
    Context for stopping Nav2 hook.

    Parameters
    ----------
    base_url : Optional[str]
        Base URL for the Nav2 system to send the stop command. If None, determined by use_sim flag.
    use_sim : bool
        Whether to run the connector in the simulator.
    api_key : str
        API key for OpenMind cloud system authentication.
    """

    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Nav2 system to send the stop command.",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to run the connector in the simulator.",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind cloud system",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def set_base_url(self) -> "StopNav2HookContext":
        """
        Set base_url based on use_sim if not explicitly provided.

        Returns
        -------
        StopNav2HookContext
            The validated context with base_url set if it was None.
        """
        if self.base_url is None:
            if self.use_sim:
                self.base_url = "https://api.openmind.com/api/core/simulation/orchestrator"
            else:
                self.base_url = "http://localhost:5000"

        return self


async def start_nav2_hook(context: Dict[str, Any]):
    """
    Hook to start Nav2 process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StartNav2HookContext(**context)
    base_url = ctx.base_url
    map_name = ctx.map_name
    api_key = ctx.api_key
    nav2_url = f"{base_url}/start/nav2"

    elevenlabs_provider: ElevenLabsTTSProvider = ElevenLabsTTSProvider()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                nav2_url,
                json={"map_name": map_name},
                headers={"Content-Type": "application/json", "x-api-key": api_key},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(f"Nav2 started successfully: {result.get('message', 'Success')}")
                    elevenlabs_provider.add_pending_message("Navigation system has started successfully.")
                    return {
                        "status": "success",
                        "message": "Nav2 process initiated",
                        "response": result,
                    }
                else:
                    try:
                        error_info = await response.json()
                    except Exception as _:
                        error_info = {"message": "Unknown error"}
                    logging.error(f"Failed to start Nav2: {error_info.get('message', 'Unknown error')}")
                    raise Exception(f"Failed to start Nav2: {error_info.get('message', 'Unknown error')}")

    except aiohttp.ClientError as e:
        logging.error(f"Error calling Nav2 API: {str(e)}")
        raise Exception(f"Error calling Nav2 API: {str(e)}")


async def stop_nav2_hook(context: Dict[str, Any]):
    """
    Hook to stop Nav2 process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StopNav2HookContext(**context)
    base_url = ctx.base_url
    api_key = ctx.api_key
    nav2_url = f"{base_url}/stop/nav2"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                nav2_url,
                headers={"Content-Type": "application/json", "x-api-key": api_key},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(f"Nav2 stopped successfully: {result.get('message', 'Success')}")
                    return {
                        "status": "success",
                        "message": "Nav2 process stopped",
                        "response": result,
                    }
                else:
                    try:
                        error_info = await response.json()
                    except Exception as _:
                        error_info = {"message": "Unknown error"}
                    logging.error(f"Failed to stop Nav2: {error_info.get('message', 'Unknown error')}")
                    raise Exception(f"Failed to stop Nav2: {error_info.get('message', 'Unknown error')}")

    except aiohttp.ClientError as e:
        logging.error(f"Error calling Nav2 stop API: {str(e)}")
        raise Exception(f"Error calling Nav2 stop API: {str(e)}")
