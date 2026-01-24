import logging
from typing import Any, Dict

import aiohttp

from hooks import Nav2Error
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider


async def start_nav2_hook(context: Dict[str, Any]):
    """
    Hook to start Nav2 process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    base_url = context.get("base_url", "http://localhost:5000")
    map_name = context.get("map_name", "map")
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
                    except Exception:
                        error_info = {"message": "Unknown error"}
                    error_msg = error_info.get("message", "Unknown error")
                    logging.error(
                        f"Failed to start Nav2: {error_msg} (status: {response.status})"
                    )
                    raise Nav2Error("Failed to start Nav2 navigation system")

    except aiohttp.ClientError as e:
        logging.error(f"Error calling Nav2 API: {e}")
        raise Nav2Error("Unable to communicate with Nav2 service") from e


async def stop_nav2_hook(context: Dict[str, Any]):
    """
    Hook to stop Nav2 process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    base_url = context.get("base_url", "http://localhost:5000")
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
                    except Exception:
                        error_info = {"message": "Unknown error"}
                    error_msg = error_info.get("message", "Unknown error")
                    logging.error(
                        f"Failed to stop Nav2: {error_msg} (status: {response.status})"
                    )
                    raise Nav2Error("Failed to stop Nav2 navigation system")

    except aiohttp.ClientError as e:
        logging.error(f"Error calling Nav2 API: {e}")
        raise Nav2Error("Unable to communicate with Nav2 service") from e
