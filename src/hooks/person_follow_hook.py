import asyncio
import logging
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider

PERSON_FOLLOW_BASE_URL = "http://localhost:8080"


class StartPersonFollowHookContext(BaseModel):
    """
    Context for starting person follow hook.

    Parameters
    ----------
    base_url : str
        Base URL for the person following system.
    enroll_timeout : float
        Time in seconds to wait for person enrollment before retrying.
    max_retries : int
        Maximum number of enrollment attempts before waiting for detection.
    """

    base_url: str = Field(
        default=PERSON_FOLLOW_BASE_URL,
        description="Base URL for the person following system",
    )
    enroll_timeout: float = Field(
        default=3.0,
        description="Time in seconds to wait for person enrollment before retrying",
    )
    max_retries: int = Field(
        default=5,
        description="Maximum number of enrollment attempts before waiting for detection",
    )

    model_config = ConfigDict(extra="allow")


class StopPersonFollowHookContext(BaseModel):
    """
    Context for stopping person follow hook.

    Parameters
    ----------
    base_url : str
        Base URL for the person following system to send the clear command.
    """

    base_url: str = Field(
        default=PERSON_FOLLOW_BASE_URL,
        description="Base URL for the person following system to send the clear command",
    )

    model_config = ConfigDict(extra="allow")


async def start_person_follow_hook(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook to start person following mode by enrolling a person to track.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StartPersonFollowHookContext(**context)

    base_url = ctx.base_url
    enroll_timeout = ctx.enroll_timeout
    max_retries = ctx.max_retries

    elevenlabs_provider = ElevenLabsTTSProvider()
    enroll_url = f"{base_url}/enroll"
    status_url = f"{base_url}/status"

    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                logging.info(
                    f"Person Follow: Enrolling (attempt {attempt + 1}/{max_retries})"
                )

                try:
                    async with session.post(
                        enroll_url,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status != 200:
                            continue
                        logging.info("Person Follow: Enroll command sent")
                except aiohttp.ClientError as e:
                    logging.warning(f"Person Follow: Enroll failed: {e}")
                    continue

                elapsed = 0.0
                while elapsed < enroll_timeout:
                    await asyncio.sleep(0.5)
                    elapsed += 0.5

                    try:
                        async with session.get(
                            status_url,
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                if status_data.get("is_tracked", False):
                                    logging.info("Person Follow: Tracking started")
                                    elevenlabs_provider.add_pending_message(
                                        "I see you! I'll follow you now."
                                    )
                                    return {
                                        "status": "success",
                                        "message": "Person enrolled and tracking",
                                        "is_tracked": True,
                                    }
                    except Exception as e:
                        logging.warning(f"Person Follow: Status poll failed: {e}")

                logging.info(
                    f"Person Follow: Attempt {attempt + 1} - not tracking, retrying"
                )

            logging.info("Person Follow: Awaiting person detection")
            elevenlabs_provider.add_pending_message(
                "Person following mode activated. Please stand in front of me."
            )
            return {
                "status": "success",
                "message": "Enrolled but awaiting person detection",
                "is_tracked": False,
            }

    except aiohttp.ClientError as e:
        logging.error(f"Person Follow: Connection error: {str(e)}")
        elevenlabs_provider.add_pending_message(
            "I couldn't connect to the person following system."
        )
        return {"status": "error", "message": f"Connection error: {str(e)}"}


async def stop_person_follow_hook(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook to stop person following mode by clearing the tracked person.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    ctx = StopPersonFollowHookContext(**context)
    base_url = ctx.base_url
    clear_url = f"{base_url}/clear"

    try:
        async with aiohttp.ClientSession() as session:
            logging.info(f"Person Follow: Calling clear at {clear_url}")

            async with session.post(
                clear_url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    logging.info("Person Follow: Cleared successfully")
                    return {"status": "success", "message": "Person tracking stopped"}
                else:
                    logging.error("Person Follow: Failed to clear")
                    return {"status": "error", "message": "Clear failed"}

    except aiohttp.ClientError as e:
        logging.error(f"Person Follow: Clear error: {str(e)}")
        return {"status": "error", "message": f"Connection error: {str(e)}"}
