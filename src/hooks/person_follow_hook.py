import asyncio
import logging
from typing import Any, Dict

import aiohttp

from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider

PERSON_FOLLOW_BASE_URL = "http://localhost:8080"


async def start_person_follow_hook(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook to start person following mode by enrolling a person to track.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    base_url = context.get("person_follow_base_url", PERSON_FOLLOW_BASE_URL)
    enroll_timeout = context.get("enroll_timeout", 3.0)
    max_retries = context.get("max_retries", 5)

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
    base_url = context.get("person_follow_base_url", PERSON_FOLLOW_BASE_URL)
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
