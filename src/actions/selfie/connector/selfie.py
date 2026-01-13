import asyncio
import logging
import time
from typing import Dict, Optional

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.selfie.interface import SelfieInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider


class SelfieConfig(ActionConfig):
    """
    Configuration for Selfie connector.

    Parameters
    ----------
    face_http_base_url : str
        Base URL for the face HTTP service.
    face_recent_sec : float
        Recency window in seconds for face detection.
    poll_ms : int
        Polling interval in milliseconds.
    timeout_sec : int
        Default timeout in seconds for operations.
    http_timeout_sec : float
        HTTP request timeout in seconds.
    """

    face_http_base_url: str = Field(
        default="http://127.0.0.1:6793",
        description="Base URL for the face HTTP service.",
    )
    face_recent_sec: float = Field(
        default=1.0,
        description="Recency window in seconds for face detection.",
    )
    poll_ms: int = Field(
        default=200,
        description="Polling interval in milliseconds.",
    )
    timeout_sec: int = Field(
        default=15,
        description="Default timeout in seconds for operations.",
    )
    http_timeout_sec: float = Field(
        default=5.0,
        description="HTTP request timeout in seconds.",
    )


class SelfieConnector(ActionConnector[SelfieConfig, SelfieInput]):
    """
    Enroll a selfie through the face HTTP service.
    """

    def __init__(self, config: SelfieConfig):
        """
        Initialize the connector.

        Parameters
        ----------
        config : SelfieConfig
            Configuration for the connector.
        """
        super().__init__(config)

        self.base_url: str = self.config.face_http_base_url

        self.recent_sec = self.config.face_recent_sec
        self.poll_ms = self.config.poll_ms
        self.default_timeout = self.config.timeout_sec
        self.http_timeout = self.config.http_timeout_sec

        self.evelenlabs_tts_provider = ElevenLabsTTSProvider()
        self.io_provider = IOProvider()

    def _write_status(self, line: str):
        """
        Make the result visible to the fuser/LLM as an input named 'SelfieStatus'.

        Parameters
        ----------
        line : str
            line: Status payload (e.g., "ok id=wendy", "failed reason=none faces=0").
        """
        try:
            self.io_provider.add_input("SelfieStatus", line, time.time())
        except Exception as e:
            logging.warning("SelfieStatus write failed: %s", e)

    def _post_json(self, path: str, body: Dict) -> Optional[Dict]:
        """
        POST JSON to the face service.

        Parameters
        ----------
        path : str
            Endpoint path (e.g., "/who", "/selfie").
        body : Dict
            Request body dict.

        Returns
        -------
        typing.Optional[Dict]
            Parsed JSON dict on success; None on error.
        """
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=body, timeout=self.http_timeout)
            return r.json()
        except Exception as e:
            logging.warning("HTTP POST %s failed (%s) body=%s", url, e, body)
            return None

    def _get_config(self) -> Dict:
        """
        Fetch current service config.

        Returns
        -------
        typing.Optional[Dict]
        """
        resp = self._post_json("/config", {"get": True}) or {}
        return resp if isinstance(resp, dict) else {}

    def _set_blur(self, on: bool) -> None:
        """
        Enable/disable blur on the service.

        Parameters
        ----------
        on : bool
            True/False
        """
        _ = self._post_json("/config", {"set": {"blur": bool(on)}})

    def _who_snapshot(self) -> Optional[Dict]:
        """
        Query current faces within the recency window.

        Returns
        -------
        typing.Optional[Dict]
            Dict with keys like "now" (list of known IDs) and "unknown_now" (int),
            or None on error.
        """
        return self._post_json("/who", {"recent_sec": self.recent_sec})

    def _wait_single_face(self, timeout_sec: int) -> bool:
        """
        Poll /who until exactly one face is visible or timeout.

        Parameters
        ----------
        timeout_sec : int
            Maximum seconds to wait (<=0 uses default_timeout).

        Returns
        -------
        bool
            True if exactly one face is detected within the timeout; False otherwise.
        """
        if timeout_sec <= 0:
            timeout_sec = self.default_timeout
        tries = max(1, int((timeout_sec * 1000) / self.poll_ms))
        for _ in range(tries):
            resp = self._who_snapshot() or {}
            now = resp.get("now") or []
            unknown_now = int(resp.get("unknown_now") or 0)
            faces = len(now) + unknown_now
            if faces == 1:
                logging.info(
                    "Selfie gate: exactly 1 face detected (now=%s, unknown=%d)",
                    now,
                    unknown_now,
                )
                return True
            time.sleep(self.poll_ms / 1000.0)
        logging.error("Selfie gate: timeout waiting for exactly 1 face.")
        return False

    async def connect(self, output_interface: SelfieInput) -> None:
        """
        Execute a single selfie enrollment attempt.

        Parameters
        ----------
        output_interface : SelfieInput
            The selfie action interface containing parameters like `id` and `timeout_sec`.
        """
        name = (output_interface.action or "").strip()
        timeout_sec = int(output_interface.timeout_sec or self.default_timeout)
        if not name:
            logging.error("Selfie requires a non-empty `id` (e.g., 'wendy').")
            self.io_provider.add_input(
                "SelfieStatus", "failed reason=bad_id", time.time()
            )
            return

        loop = asyncio.get_running_loop()

        cfg = await loop.run_in_executor(None, self._get_config)
        orig_blur = bool(((cfg or {}).get("config") or {}).get("blur", True))
        await loop.run_in_executor(None, self._set_blur, False)

        try:
            ok = await loop.run_in_executor(None, self._wait_single_face, timeout_sec)
            if not ok:
                snapshot = await loop.run_in_executor(None, self._who_snapshot) or {}
                now = snapshot.get("now") or []
                unknown_now = int(snapshot.get("unknown_now") or 0)
                faces = len(now) + unknown_now
                reason = "none" if faces == 0 else "multiple"
                logging.info("[Selfie] Gating failed: %s (faces=%d)", reason, faces)
                self.io_provider.add_input(
                    "SelfieStatus",
                    f"failed reason={reason} faces={faces}",
                    time.time(),
                )
                self.evelenlabs_tts_provider.add_pending_message(
                    f"Woof! Woof! I saw {faces} faces. Please make sure only your face is visible and try again."
                )
                return

            resp = await loop.run_in_executor(
                None, self._post_json, "/selfie", {"id": name}
            )
            if not (isinstance(resp, dict) and resp.get("ok")):
                logging.error("[Selfie] /selfie failed or returned non-ok: %s", resp)
                self.io_provider.add_input(
                    "SelfieStatus", "failed reason=service", time.time()
                )
                self.evelenlabs_tts_provider.add_pending_message(
                    "Woof! Woof! I couldn't see you clearly. Please try again."
                )
                return

            logging.info("[Selfie] Enrolled selfie for '%s' successfully.", name)
            self.io_provider.add_input("SelfieStatus", f"ok id={name}", time.time())
            self.evelenlabs_tts_provider.add_pending_message(
                f"Woof! Woof! I remember you, {name}! You are now enrolled."
            )

        finally:
            await loop.run_in_executor(None, self._set_blur, orig_blur)
