import logging

import aiohttp
from pydantic import Field

from backgrounds.base import Background, BackgroundConfig


class UnitreeGo2PatrolConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Patrol Background.
    """

    base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the patrol control API",
    )


class UnitreeGo2Patrol(Background[UnitreeGo2PatrolConfig]):
    """
    Background task for patrolling with Unitree Go2 robot.
    """

    def __init__(self, config: UnitreeGo2PatrolConfig):
        """
        Initialize Patrol background task with configuration.

        Parameters
        ----------
        config : UnitreeGo2PatrolConfig
            Configuration for the Unitree Go2 Patrol background task, including patrol parameters and options.
        """
        super().__init__(config)
        logging.info("Initialized Unitree Go2 Patrol Background Task")

    async def start_patrol(self) -> None:
        """
        Start the patrol behavior.
        """
        logging.info("Starting Unitree Go2 Patrol")
        url = f"{self.config.base_url}/patrol/start"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    logging.info(f"Patrol started successfully: {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to start patrol: {e}")
            raise

    async def stop_patrol(self) -> None:
        """
        Stop the patrol behavior.
        """
        logging.info("Stopping Unitree Go2 Patrol")
        url = f"{self.config.base_url}/patrol/stop"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    logging.info(f"Patrol stopped successfully: {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to stop patrol: {e}")
            raise

    async def pause_patrol(self) -> None:
        """
        Pause the patrol behavior.
        """
        logging.info("Pausing Unitree Go2 Patrol")
        url = f"{self.config.base_url}/patrol/pause"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    logging.info(f"Patrol paused successfully: {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to pause patrol: {e}")
            raise

    async def resume_patrol(self) -> None:
        """
        Resume the patrol behavior.
        """
        logging.info("Resuming Unitree Go2 Patrol")
        url = f"{self.config.base_url}/patrol/resume"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    logging.info(f"Patrol resumed successfully: {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to resume patrol: {e}")
            raise
