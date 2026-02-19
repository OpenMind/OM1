import logging
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class StartNav2HookContext(BaseModel):
    """
    Context for starting Nav2 hook.

    Parameters
    ----------
    base_url : str
        Base URL for the Nav2 system.
    map_name : str
        Name of the map to use for navigation.
    """

    base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the Nav2 system",
    )
    map_name: str = Field(
        default="map",
        description="Name of the map to use for navigation",
    )

    model_config = ConfigDict(extra="allow")


class StopNav2HookContext(BaseModel):
    """
    Context for stopping Nav2 hook.

    Parameters
    ----------
    base_url : str
        Base URL for the Nav2 system to send the stop command.
    """

    base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the Nav2 system to send the stop command",
    )

    model_config = ConfigDict(extra="allow")


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
    nav2_url = f"{base_url}/start/nav2"

    elevenlabs_provider: ElevenLabsTTSProvider = ElevenLabsTTSProvider()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                nav2_url,
                json={"map_name": map_name},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(
                        f"Nav2 started successfully: {result.get('message', 'Success')}"
                    )
                    elevenlabs_provider.add_pending_message(
                        "Navigation system has started successfully."
                    )
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
                    logging.error(
                        f"Failed to start Nav2: {error_info.get('message', 'Unknown error')}"
                    )
                    raise Exception(
                        f"Failed to start Nav2: {error_info.get('message', 'Unknown error')}"
                    )

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
    nav2_url = f"{base_url}/stop/nav2"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                nav2_url,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(
                        f"Nav2 stopped successfully: {result.get('message', 'Success')}"
                    )
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
                    logging.error(
                        f"Failed to stop Nav2: {error_info.get('message', 'Unknown error')}"
                    )
                    raise Exception(
                        f"Failed to stop Nav2: {error_info.get('message', 'Unknown error')}"
                    )

    except aiohttp.ClientError as e:
        logging.error(f"Error calling Nav2 stop API: {str(e)}")
        raise Exception(f"Error calling Nav2 stop API: {str(e)}")
