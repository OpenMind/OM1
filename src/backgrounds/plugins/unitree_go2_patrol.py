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

    This background task manages the patrol behavior of a Unitree Go2 robot.
    It can be configured to follow predefined waypoints, perform area coverage,
    or execute specific patrol patterns. The task continuously monitors the
    robot's state and environment to ensure safe and efficient patrolling.
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

        This method initiates the patrol routine, which may involve navigating
        through waypoints, performing area coverage, or executing specific
        patrol patterns. The method continuously monitors the robot's state and
        environment to ensure safe and efficient patrolling.
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

        This method safely halts the patrol routine, ensuring that the robot
        comes to a stop and any ongoing navigation or movement commands are
        terminated. It may also perform any necessary cleanup or state resets
        related to the patrol behavior.
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

        This method temporarily pauses the patrol routine, allowing the robot to
        halt its movement while maintaining its current state. The patrol can be
        resumed later without losing progress or state information.
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

        This method resumes the patrol routine after it has been paused, allowing
        the robot to continue its patrolling activities from where it left off.
        It ensures that any necessary state information is maintained for a
        seamless continuation of the patrol.
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
