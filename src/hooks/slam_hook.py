import logging
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


class StartSlamHookContext(BaseModel):
    """
    Context for starting SLAM hook.

    Parameters
    ----------
    base_url : str
        Base URL for the SLAM system.
    """

    base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the SLAM system",
    )

    model_config = ConfigDict(extra="allow")


class StopSlamHookContext(BaseModel):
    """
    Context for stopping SLAM hook.

    Parameters
    ----------
    base_url : str
        Base URL for the SLAM system to send the stop command.
    map_name : str
        Name of the map to save before stopping SLAM.
    """

    base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the SLAM system to send the stop command",
    )
    map_name: str = Field(
        default="map",
        description="Name of the map to save before stopping SLAM",
    )

    model_config = ConfigDict(extra="allow")


async def start_slam_hook(context: Dict[str, Any]):
    """
    Hook to start SLAM process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StartSlamHookContext(**context)
    base_url = ctx.base_url
    slam_url = f"{base_url}/start/slam"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                slam_url,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(
                        f"SLAM started successfully: {result.get('message', 'Success')}"
                    )
                    return {
                        "status": "success",
                        "message": "SLAM process initiated",
                        "response": result,
                    }
                else:
                    try:
                        error_info = await response.json()
                    except Exception as _:
                        error_info = {"message": "Unknown error"}
                    logging.error(
                        f"Failed to start SLAM: {error_info.get('message', 'Unknown error')}"
                    )
                    raise Exception(
                        f"Failed to start SLAM: {error_info.get('message', 'Unknown error')}"
                    )

    except aiohttp.ClientError as e:
        logging.error(f"Error calling SLAM API: {str(e)}")
        raise Exception(f"Error calling SLAM API: {str(e)}")


async def stop_slam_hook(context: Dict[str, Any]):
    """
    Hook to stop SLAM process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StopSlamHookContext(**context)
    base_url = ctx.base_url
    map_name = ctx.map_name

    save_slam_map_url = f"{base_url}/maps/save"
    stop_slam_url = f"{base_url}/stop/slam"

    elevenlabs_provider: ElevenLabsTTSProvider = ElevenLabsTTSProvider()

    try:
        async with aiohttp.ClientSession() as session:
            # Save the SLAM map before stopping
            async with session.post(
                save_slam_map_url,
                json={"map_name": map_name},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as save_response:

                if save_response.status == 200:
                    save_result = await save_response.json()
                    logging.info(
                        f"SLAM map saved successfully: {save_result.get('message', 'Success')}"
                    )
                    elevenlabs_provider.add_pending_message(
                        "Map has been saved successfully."
                    )
                else:
                    try:
                        error_info = await save_response.json()
                    except Exception as _:
                        error_info = {"message": "Unknown error"}
                    logging.error(
                        f"Failed to save SLAM map: {error_info.get('message', 'Unknown error')}"
                    )
                    raise Exception(
                        f"Failed to save SLAM map: {error_info.get('message', 'Unknown error')}"
                    )

            # Stop the SLAM process
            async with session.post(
                stop_slam_url,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    logging.info(
                        f"SLAM stopped successfully: {result.get('message', 'Success')}"
                    )
                    return {
                        "status": "success",
                        "message": "SLAM process stopped",
                        "response": result,
                    }
                else:
                    try:
                        error_info = await response.json()
                    except Exception as _:
                        error_info = {"message": "Unknown error"}
                    logging.error(
                        f"Failed to stop SLAM: {error_info.get('message', 'Unknown error')}"
                    )
                    raise Exception(
                        f"Failed to stop SLAM: {error_info.get('message', 'Unknown error')}"
                    )

    except aiohttp.ClientError as e:
        logging.error(f"Error calling SLAM API: {str(e)}")
        raise Exception(f"Error calling SLAM API: {str(e)}")
