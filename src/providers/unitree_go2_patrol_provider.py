import base64
import io
import logging

import aiohttp

from .singleton import singleton


@singleton
class UnitreeGo2PatrolProvider:
    """
    Provider for controlling Unitree Go2 patrol behavior via HTTP API.

    This class implements a singleton pattern to manage patrol operations
    including start, stop, pause, and resume functionality.
    """

    def __init__(
        self,
        api_key: str = "",
        patrol_base_url: str = "http://localhost:5000",
        face_presence_base_url: str = "http://localhost:6793",
        patrol_image_report_base_url: str = "https://api.openmind.com",
    ):
        """
        Initialize the Patrol Provider.

        Parameters
        ----------
        api_key : str, optional
            API key for OpenMind patrol upload endpoint (default is "").
        patrol_base_url : str, optional
            Base URL for the patrol control API (default is "http://localhost:5000").
        face_presence_base_url : str, optional
            Base URL for the patrol report API (default is "http://localhost:6793").
        patrol_image_report_base_url : str, optional
            URL for reporting patrol images to OpenMind API (default is "https://api.openmind.com").
        """
        self.patrol_base_url = patrol_base_url
        self.face_presence_base_url = face_presence_base_url
        self.patrol_image_report_base_url = patrol_image_report_base_url
        self.api_key = api_key
        logging.info(
            f"Initialized Patrol Provider with patrol base URL: {patrol_base_url}, "
            f"face presence URL: {face_presence_base_url}"
        )

    async def start_patrol(self) -> None:
        """
        Start the patrol behavior.

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        logging.info("Starting patrol")
        url = f"{self.patrol_base_url}/patrol/start"
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

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        logging.info("Stopping patrol")
        url = f"{self.patrol_base_url}/patrol/stop"
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

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        logging.info("Pausing patrol")
        url = f"{self.patrol_base_url}/patrol/pause"
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

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        logging.info("Resuming patrol")
        url = f"{self.patrol_base_url}/patrol/resume"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    response.raise_for_status()
                    logging.info(f"Patrol resumed successfully: {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to resume patrol: {e}")
            raise

    async def get_report(self, recent_sec: int = 3) -> dict:
        """
        Get the current patrol report.

        Parameters
        ----------
        recent_sec : int, optional
            Time window in seconds for recent face presence data (default is 3 seconds).

        Returns
        -------
        dict
            The patrol report data.

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        url = f"{self.face_presence_base_url}/who"
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                async with session.post(url, json={"recent_sec": recent_sec}, headers=headers) as response:
                    response.raise_for_status()
                    report_data = await response.json()
                    return report_data
        except aiohttp.ClientError as e:
            logging.error(f"Failed to get patrol report: {e}")
            raise

    async def upload_patrol_image(self, image_base64: str, description: str = "") -> dict:
        """
        Upload patrol image to OpenMind API.

        Parameters
        ----------
        image_base64 : str
            Base64 encoded image data.
        description : str, optional
            Description of the patrol image (default is "").

        Returns
        -------
        dict
            The response data from the upload endpoint.

        Raises
        ------
        aiohttp.ClientError
            If the HTTP request fails.
        """
        if not self.api_key:
            logging.warning("No API key provided, skipping patrol image upload")
            return {}

        logging.info("Uploading patrol image")
        url = self.patrol_image_report_base_url

        try:
            image_data = base64.b64decode(image_base64)
            image_file = io.BytesIO(image_data)

            form_data = aiohttp.FormData()
            form_data.add_field("image", image_file, filename="patrol_image.jpg", content_type="image/jpeg")
            form_data.add_field("description", description)

            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{url}/api/core/patrol/upload", data=form_data, headers=headers) as response:
                    response.raise_for_status()
                    result_data = await response.json()
                    logging.info(f"Patrol image uploaded successfully: {response.status}")
                    return result_data
        except aiohttp.ClientError as e:
            logging.error(f"Failed to upload patrol image: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing patrol image upload: {e}")
            raise
