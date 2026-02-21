import json
import logging
import threading
import time
from typing import Callable, Dict, Optional

import requests

from .singleton import singleton


@singleton
class UbtechASRProvider:
    """
    Singleton class to handle ASR (Automatic Speech Recognition) for Ubtech robots.
    """

    _instance = None

    @staticmethod
    def get_instance():
        """Returns the singleton instance of the provider."""
        return UbtechASRProvider._instance

    def __init__(self, robot_ip: str, language_code: str = "en"):
        """
        Initialize the Ubtech ASR Provider.

        Parameters
        ----------
        robot_ip : str
            The IP address of the robot.
        language_code : str
            The language code for ASR. Defaults to "en".
        """
        UbtechASRProvider._instance = self

        self.robot_ip = robot_ip
        self.language = language_code

        self.basic_url = f"http://{self.robot_ip}:9090/v1/"
        self.headers = {"Content-Type": "application/json"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.running = False
        self.paused = False
        self.just_resumed = False
        self._thread: Optional[threading.Thread] = None
        self._message_callback: Optional[Callable] = None
        self._set_robot_language(language_code)

    def register_message_callback(self, cb: Optional[Callable]):
        """
        Register a callback to process recognized ASR messages.

        Parameters
        ----------
        cb : Optional[callable]
            The callback function to process recognized ASR messages.
        """
        self._message_callback = cb

    def start(self):
        """Start the ASR provider and its background thread."""
        if self.running:
            return

        logging.info("Starting UbtechASRProvider background thread...")
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the ASR provider and its background thread."""
        if not self.running:
            return

        logging.info("Stopping UbtechASRProvider background thread...")
        self.running = False
        self._stop_voice_iat()

        if self._thread:
            self._thread.join(timeout=3)

        logging.info("UbtechASRProvider stopped.")

    def pause(self):
        """Pause the ASR provider to stop listening for new utterances."""
        logging.debug("Pausing UbtechASRProvider")
        self.paused = True

    def resume(self):
        """Resume the ASR provider after being paused."""
        logging.debug("Resuming UbtechASRProvider")
        self.paused = False
        self.just_resumed = True

    def _run(self):
        while self.running:
            if self.just_resumed:
                logging.debug(
                    "UbtechASRProvider: Just resumed, adding a 1.0s delay before listening."
                )
                time.sleep(1.0)
                self.just_resumed = False

            if self.paused:
                time.sleep(0.1)
                continue

            text = None
            try:
                logging.debug(
                    "UbtechASRProvider: _run loop iteration, attempting to get utterance."
                )
                text = self._get_single_utterance()

                if text:
                    logging.debug(
                        f"UbtechASRProvider: Successfully got utterance: '{text}'"
                    )
                    if self._message_callback:
                        logging.debug(
                            f"UbtechASRProvider: Calling message callback with: '{text}'"
                        )
                        self._message_callback(text)

            except requests.RequestException as e:
                logging.error(
                    f"UbtechASRProvider: RequestException during _get_single_utterance: {e}"
                )
            except Exception as e:
                logging.error(
                    f"UbtechASRProvider: Unexpected error in _run's try block: {e}",
                    exc_info=True,
                )
            finally:
                logging.debug(
                    "UbtechASRProvider: Reached finally block in _run, ensuring ASR is paused."
                )
                self.pause()

                if text:
                    logging.info(
                        f"ASR got text: '{text}'. Pausing. Provider now sleeping for 2.0s to allow system processing."
                    )
                    time.sleep(2.0)
                else:
                    logging.debug(
                        "UbtechASRProvider: No text obtained or error occurred, sleeping briefly before allowing resume."
                    )
                    time.sleep(0.5)

    def _get_single_utterance(self) -> Optional[str]:
        ts = int(time.time())
        logging.debug(
            f"UbtechASRProvider: _get_single_utterance called, timestamp: {ts}"
        )

        try:
            # PREEMPTIVE STOP: Ensure any previous session is cleared before starting a new one.
            logging.info(
                "UbtechASRProvider: Preemptively stopping any existing ASR session."
            )
            self._stop_voice_iat()
            time.sleep(0.1)

            if not self._start_voice_iat(ts):
                logging.warning(
                    "UbtechASRProvider: _start_voice_iat failed or returned False. ASR session not started."
                )
                return None

            logging.debug(
                "UbtechASRProvider: ASR session started, pausing for 0.2s before polling."
            )
            time.sleep(0.2)

            for i in range(100):
                if not self.running:
                    logging.debug(
                        "UbtechASRProvider: Not running, exiting _get_single_utterance loop."
                    )
                    return None
                res = self._get_voice_iat()
                logging.debug(
                    f"UbtechASRProvider: _get_voice_iat (attempt {i+1}) returned: {res}"
                )
                if res.get("status") == "idle" and res.get("timestamp") == ts:
                    if not res.get("data") or res.get("code") != 0:
                        logging.debug(
                            "UbtechASRProvider: Idle status but no data or error code."
                        )
                        return None
                    words = res.get("data", {}).get("text", {}).get("ws", [])
                    processed_text = (
                        "".join(w["cw"][0]["w"] for w in words).strip().lower()
                        if words
                        else None
                    )
                    logging.debug(
                        f"UbtechASRProvider: Processed text from idle status: '{processed_text}'"
                    )
                    return processed_text
                time.sleep(0.1)
            logging.debug(
                "UbtechASRProvider: _get_single_utterance loop finished without idle status (timeout)."
            )
            return None
        finally:
            logging.debug(
                "UbtechASRProvider: Ensuring current voice iat session is stopped (post-attempt)."
            )
            self._stop_voice_iat()

    def _set_robot_language(self, lang_code: str):
        try:
            logging.info(f"Setting robot language to: {lang_code}")
            self.session.put(
                f"{self.basic_url}system/language",
                data=json.dumps({"language": lang_code}),
                timeout=3,
            )
        except requests.RequestException as e:
            logging.error(f"UbtechASRProvider: Failed to set robot language: {e}")

    def _start_voice_iat(self, ts: int) -> bool:
        if not self.robot_ip:
            logging.error("Robot IP not set, cannot start ASR session.")
            return False
        try:
            data = {"text": "", "timestamp": ts, "lang": self.language}
            logging.debug(
                f"Starting ASR session with timestamp {ts} and language {self.language}. Payload: {data}"
            )
            res = self.session.put(f"{self.basic_url}voice/iat", json=data, timeout=3)
            res.raise_for_status()
            response_json = res.json()
            logging.debug(
                f"UbtechASRProvider: _start_voice_iat response: {response_json}"
            )
            return response_json.get("code") == 0
        except requests.RequestException as e:
            logging.error(f"UbtechASRProvider: _start_voice_iat request failed: {e}")
            return False

    def _stop_voice_iat(self):
        max_retries = 3
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                logging.debug(
                    f"UbtechASRProvider: Attempting to stop voice iat (attempt {attempt + 1}/{max_retries})."
                )
                response = self.session.delete(f"{self.basic_url}voice/iat", timeout=2)
                response.raise_for_status()
                logging.debug("UbtechASRProvider: _stop_voice_iat request successful.")
                return
            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    logging.warning(
                        f"UbtechASRProvider: _stop_voice_iat failed with {e.response.status_code} (attempt {attempt + 1}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(
                        f"UbtechASRProvider: _stop_voice_iat failed with HTTPError (attempt {attempt + 1}): {e}"
                    )
                    return
            except requests.RequestException as e:
                logging.error(
                    f"UbtechASRProvider: _stop_voice_iat request failed (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return
        logging.error(
            f"UbtechASRProvider: _stop_voice_iat failed after {max_retries} retries."
        )

    def _get_voice_iat(self) -> Dict:
        max_retries = 3
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                logging.debug(
                    f"UbtechASRProvider: Attempting to get voice iat (attempt {attempt + 1}/{max_retries})."
                )
                res = self.session.get(f"{self.basic_url}voice/iat", timeout=3)
                res.raise_for_status()
                response_json = res.json()
                logging.debug(
                    f"UbtechASRProvider: _get_voice_iat raw response: {response_json}"
                )

                if (
                    response_json.get("data")
                    and isinstance(response_json["data"], str)
                    and response_json.get("code") == 0
                ):
                    cleaned_data_str = response_json["data"].strip().rstrip("\x00")
                    logging.debug(
                        f"UbtechASRProvider: _get_voice_iat cleaned data string: '{cleaned_data_str}'"
                    )
                    if cleaned_data_str:
                        try:
                            response_json["data"] = json.loads(cleaned_data_str)
                        except json.JSONDecodeError:
                            logging.error(
                                f"UbtechASRProvider: Failed to decode JSON from data string: '{cleaned_data_str}'"
                            )
                return response_json
            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    logging.warning(
                        f"UbtechASRProvider: _get_voice_iat failed with {e.response.status_code} (attempt {attempt + 1}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(
                        f"UbtechASRProvider: _get_voice_iat failed with HTTPError (attempt {attempt + 1}): {e}"
                    )
                    return {
                        "code": -1,
                        "message": str(e),
                        "data": None,
                        "status": "error",
                    }
            except requests.RequestException as e:
                logging.error(
                    f"UbtechASRProvider: _get_voice_iat request failed (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"code": -1, "message": str(e), "data": None, "status": "error"}
            except json.JSONDecodeError as e:
                logging.error(
                    f"UbtechASRProvider: _get_voice_iat failed to decode main JSON response (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {
                    "code": -1,
                    "message": f"JSONDecodeError: {e}",
                    "data": None,
                    "status": "error",
                }

        logging.error(
            f"UbtechASRProvider: _get_voice_iat failed after {max_retries} retries."
        )
        return {
            "code": -1,
            "message": "Max retries exceeded",
            "data": None,
            "status": "error",
        }
