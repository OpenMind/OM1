import json
import logging

import requests


class UbTtsProvider:
    """
    Provider for the Ubtech Text-to-Speech (TTS) service.

    This class handles communication with the Ubtech TTS service API, providing
    methods to send TTS commands and query the status of TTS tasks. It manages
    HTTP requests to the TTS service endpoint and handles error responses.

    Attributes
    ----------
    tts_url : str
        Base URL of the TTS service.
    headers : dict
        HTTP headers used for requests (e.g. Content-Type).
    """

    def __init__(self, url: str):
        """
        Initialize the Ubtech TTS Provider.

        Parameters
        ----------
        url : str
            The base URL endpoint for the Ubtech TTS service API.
        """
        self.tts_url = url
        self.headers = {"Content-Type": "application/json"}
        logging.info(f"Ubtech TTS Provider initialized for URL: {self.tts_url}")

    def speak(self, tts: str, interrupt: bool = True, timestamp: int = 0) -> bool:
        """
        Send a text-to-speech request to the TTS service.

        This method sends a PUT request to the TTS service with the specified
        text, interrupt flag, and timestamp. The request includes a timeout
        of 5 seconds and handles network errors gracefully.

        Parameters
        ----------
        tts : str
            The text content to be converted to speech.
        interrupt : bool, optional
            Whether to interrupt any currently playing TTS output. Defaults to True.
        timestamp : int, optional
            Timestamp identifier for the TTS task. Defaults to 0.

        Returns
        -------
        bool
            True if the TTS request was successfully sent and accepted by the service
            (response code is 0), False otherwise.

        Notes
        -----
        Network errors and HTTP exceptions are caught and logged, with the method
        returning False to indicate failure.
        """
        payload = {"tts": tts, "interrupt": interrupt, "timestamp": timestamp}
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
        Get the current status of a specific TTS task.

        This method queries the TTS service for the status of a task identified
        by the given timestamp. The request includes a timeout of 2 seconds.

        Parameters
        ----------
        timestamp : int
            The timestamp identifier of the TTS task to query.

        Returns
        -------
        str
            The status of the TTS task. Possible values:
            - 'build': Task is being built/prepared
            - 'wait': Task is waiting in queue
            - 'run': Task is currently running
            - 'idle': Task is idle/not active
            - 'error': An error occurred or the task was not found

        Notes
        -----
        Network errors and HTTP exceptions are caught silently, with the method
        returning 'error' to indicate failure.
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
