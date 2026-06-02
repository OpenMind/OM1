import logging
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from providers.unitree_go2_patrol_provider import UnitreeGo2PatrolProvider


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


class UnitreeGo2PatrolHookContext(BaseModel):
    """
    Context for Unitree Go2 patrol hook.

    Parameters
    ----------
    patrol_base_url : str
        Base URL for the patrol control API.
    face_presence_base_url : str
        Base URL for the face presence API.
    patrol_image_report_base_url : str
        URL for reporting patrol data to OpenMind API.
    api_key : str
        API key for OpenMind patrol upload endpoint.
    """

    patrol_base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the patrol control API",
    )
    face_presence_base_url: str = Field(
        default="http://127.0.0.1:6793",
        description="Base URL for the face presence API",
    )
    patrol_image_report_base_url: str = Field(
        default="https://api.openmind.com",
        description="URL for reporting patrol data to OpenMind API",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind patrol upload endpoint",
    )


async def start_unitree_go2_patrol_hook(context: Dict[str, Any]):
    """
    Hook to start Unitree Go2 patrol process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    logging.info("Starting Unitree Go2 patrol with context: %s", context)
    ctx = UnitreeGo2PatrolHookContext(**context)

    patrol_base_url = ctx.patrol_base_url
    face_presence_base_url = ctx.face_presence_base_url
    patrol_image_report_base_url = ctx.patrol_image_report_base_url
    api_key = ctx.api_key

    patrol_provider = UnitreeGo2PatrolProvider(
        api_key=api_key,
        patrol_base_url=patrol_base_url,
        face_presence_base_url=face_presence_base_url,
        patrol_image_report_base_url=patrol_image_report_base_url,
    )
    try:
        await patrol_provider.start_patrol()
        logging.info("Unitree Go2 patrol started successfully")

        return True
    except Exception:
        logging.exception("Error in starting Unitree Go2 patrol")
        return False


async def stop_unitree_go2_patrol_hook(context: Dict[str, Any]):
    """
    Hook to stop Unitree Go2 patrol process.

    Parameters
    ----------
    context : Dict[str, Any]
        Context dictionary containing configuration parameters.
    """
    logging.info("Stopping Unitree Go2 patrol with context: %s", context)
    ctx = UnitreeGo2PatrolHookContext(**context)

    patrol_base_url = ctx.patrol_base_url
    face_presence_base_url = ctx.face_presence_base_url
    patrol_image_report_base_url = ctx.patrol_image_report_base_url
    api_key = ctx.api_key

    patrol_provider = UnitreeGo2PatrolProvider(
        api_key=api_key,
        patrol_base_url=patrol_base_url,
        face_presence_base_url=face_presence_base_url,
        patrol_image_report_base_url=patrol_image_report_base_url,
    )

    try:
        await patrol_provider.stop_patrol()
        logging.info("Unitree Go2 patrol stopped successfully")

        return True
    except Exception:
        logging.exception("Error in stopping Unitree Go2 patrol")
        return False
