import asyncio
import logging
import time
import typing

import requests

from actions.base import ActionConfig, ActionConnector
from actions.selfie.interface import SelfieInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider

_JSON = typing.Dict[str, typing.Any]


class SelfieConnector(ActionConnector[SelfieInput]):
    """
    Enroll a selfie through the face HTTP service :
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the connector

        Parameters
        ----------
        config : ActionConfig
            Configuration for the connector.
        """
        super().__init__(config)

        self.base_url: str = getattr(
            self.config, "face_http_base_url", "http://127.0.0.1:6793"
        )

        self.recent_sec: float = float(getattr(self.config, "face_recent_sec", 1.0))
        self.poll_ms: int = int(getattr(self.config, "poll_ms", 200))
        self.default_timeout: int = int(getattr(self.config, "timeout_sec", 15))
        self.http_timeout: float = float(getattr(self.config, "http_timeout_sec", 5.0))

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

    def _post_json(self, path: str, body: _JSON) -> typing.Optional[_JSON]:
        """
        POST JSON to the face service.

        Parameters
        ----------
        path : str
            Endpoint path (e.g., "/who", "/selfie").
        body : _JSON
            Request body dict.

        Returns
        -------
        typing.Optional[_JSON]
            Parsed JSON dict on success; None on error.
        """
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=body, timeout=self.http_timeout)
            return r.json()
        except Exception as e:
            logging.warning("HTTP POST %s failed (%s) body=%s", url, e, body)
            return None

    def _get_config(self) -> _JSON:
        """
        Fetch current service config.

        Returns
        -------
        typing.Optional[_JSON]
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

    def _who_snapshot(self) -> typing.Optional[_JSON]:
        """
        Query current faces within the recency window.

        Returns
        -------
        typing.Optional[_JSON]
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
