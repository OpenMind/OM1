import json
import logging
from concurrent.futures import ThreadPoolExecutor

import requests


class UbTtsProvider:
    """
    Provider for the Ubtech Text-to-Speech (TTS) service.
    """

    def __init__(self, url: str):
        """
        Initialize the Ubtech TTS Provider.

        Parameters
        ----------
        url : str
            The URL of the Ubtech TTS service.
        """
        self.tts_url = url
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.headers = {"Content-Type": "application/json"}

        logging.info(f"Ubtech TTS Provider initialized for URL: {self.tts_url}")

    def start(self):
        """Placeholder start method for compatibility."""
        logging.info("Ubtech TTS Provider started.")

    def adding_pending_message(
        self, message: str, interrupt: bool = True, timestamp: int = 0
    ):
        """
        Add a pending TTS message to be processed asynchronously.

        Parameters
        ----------
        message : str
            The text message to be converted to speech.
        interrupt : bool
            Whether to interrupt current TTS playback.
        timestamp : int
            A timestamp to identify the TTS request.
        """
        self.executor.submit(self._speak_workder, message, interrupt, timestamp)

    def stop(self):
        """
        Stop the TeleopsStatusProvider and clean up resources.
        """
        self.executor.shutdown(wait=True)

    def _speak_workder(
        self, message: str, interrupt: bool = True, timestamp: int = 0
    ) -> bool:
        """
        Worker function to send TTS command to Ubtech TTS service.

        Parameters
        ----------
        message : str
            The text to be converted to speech.
        interrupt : bool
            Whether to interrupt current TTS playback.
        timestamp : int
            A timestamp to identify the TTS request.

        Returns
        -------
        bool
            True if the TTS command was successfully sent, False otherwise.
        """
        payload = {"tts": message, "interrupt": interrupt, "timestamp": timestamp}
        try:
            response = requests.put(
                url=self.tts_url,
                data=json.dumps(payload),
                headers=self.headers,
                timeout=5,
            )
            response.raise_for_status()
            res = response.json()
            return res.get("code") == 0
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send TTS command: {e}")
            return False

    def get_tts_status(self, timestamp: int) -> str:
        """
        Gets the status of a specific TTS task.
        Possible statuses: 'build', 'wait', 'run', 'idle'.
        """
        try:
            params = {"timestamp": timestamp}
            response = requests.get(
                url=self.tts_url, headers=self.headers, params=params, timeout=2
            )
            res = response.json()
            if res.get("code") == 0:
                return res.get("status", "error")
            return "error"
        except requests.exceptions.RequestException:
            return "error"
