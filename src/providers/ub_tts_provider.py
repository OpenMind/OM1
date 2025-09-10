import json
import logging

import requests


class UbTtsProvider:
    """
    Provider for the Ubtech Text-to-Speech (TTS) service.
    """

    def __init__(self, url: str):
        self.tts_url = url
        self.headers = {"Content-Type": "application/json"}
        logging.info(f"Ubtech TTS Provider initialized for URL: {self.tts_url}")

    def speak(self, tts: str, interrupt: bool = True, timestamp: int = 0) -> bool:
        """Sends a request to the TTS service. Returns True on success."""
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
