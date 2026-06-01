"""
Forget Last connector for OM1.

Undoes the most recent /selfie enrollment by calling /gallery/forget_last
on the face API. Used when the WRONG PERSON was captured (samples are
wrong, not just the label).

The connector is stateless — the face API enforces the 60s TTL.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.forget_last.interface import ForgetLastInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider


class ForgetLastConfig(ActionConfig):
    """
    Configuration for ForgetLast connector.

    Parameters
    ----------
    face_http_base_url : str
        Base URL for the face HTTP service.
    http_timeout_sec : float
        HTTP request timeout in seconds.
    """

    face_http_base_url: str = Field(
        default="http://127.0.0.1:6793",
        description="Base URL for the face HTTP service.",
    )
    http_timeout_sec: float = Field(
        default=5.0,
        description="HTTP request timeout in seconds.",
    )


class ForgetLastConnector(ActionConnector[ForgetLastConfig, ForgetLastInput]):
    """Undo the most recent enrollment via /gallery/forget_last."""

    def __init__(self, config: ForgetLastConfig):
        super().__init__(config)
        self.base_url = self.config.face_http_base_url
        self.http_timeout = self.config.http_timeout_sec
        self.elevenlabs_tts_provider = ElevenLabsTTSProvider()
        self.io_provider = IOProvider()

    # -------- HTTP helpers --------

    def _post_json(self, path: str, body: Dict) -> Optional[Dict]:
        """POST JSON to the face service. Returns parsed dict or None on error."""
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=body, timeout=self.http_timeout)
            return r.json()
        except Exception as e:
            logging.warning("HTTP POST %s failed (%s) body=%s", url, e, body)
            return None

    # -------- Output helpers --------

    def _write_status(self, line: str) -> None:
        """
        Surface the result to the LLM via the shared SelfieStatus channel.

        All face-memory actions (selfie, correct_identity, forget_last) write
        to the single 'SelfieStatus' io_provider key so the existing
        SelfieStatus input plugin picks them all up. The LLM disambiguates
        by reading the `result=...` prefix in the line.
        """
        try:
            self.io_provider.add_input("SelfieStatus", line, time.time())
        except Exception as e:
            logging.warning("SelfieStatus write failed: %s", e)

    def _speak(self, message: Optional[str]) -> None:
        """Queue a TTS message. No-op if empty."""
        if not message:
            return
        try:
            self.elevenlabs_tts_provider.add_pending_message(message)
        except Exception as e:
            logging.warning("TTS queue failed: %s", e)

    # -------- Main entry point --------

    async def connect(self, output_interface: ForgetLastInput) -> None:
        """Execute a single undo of the most recent enrollment."""
        body: Dict = {}
        id_check = getattr(output_interface, "id", None)
        if id_check:
            body["id"] = str(id_check).strip().lower()

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, self._post_json, "/gallery/forget_last", body)

        self._dispatch_response(resp, requested_id=id_check)

    def _dispatch_response(self, resp: Optional[Dict], requested_id: Optional[str]) -> None:
        """Map the API response to status + TTS."""
        if resp is None:
            self._write_status("result=network_error")
            self._speak("I couldn't undo that.")
            logging.error("[ForgetLast] network error")
            return

        if resp.get("ok"):
            forgotten_id = str(resp.get("id", ""))
            deleted = int(resp.get("files_deleted", 0))
            identity_removed = bool(resp.get("identity_removed"))
            self._write_status(
                f"result=success id={forgotten_id} files_deleted={deleted} "
                f"identity_removed={'true' if identity_removed else 'false'}"
            )
            self._speak("OK, I've forgotten that one. Let's try again.")
            logging.info(
                "[ForgetLast] ok id=%s deleted=%d identity_removed=%s",
                forgotten_id,
                deleted,
                identity_removed,
            )
            return

        err = str(resp.get("error", "unknown"))

        if err in ("no_recent_enrollment", "stale_enrollment"):
            self._write_status(f"result={err}")
            self._speak("There's nothing recent for me to forget.")
            logging.info("[ForgetLast] %s", err)
            return

        if err == "id_mismatch":
            detail = str(resp.get("detail", ""))
            requested = (requested_id or "").strip().lower()
            self._write_status(f"result=id_mismatch requested={requested} detail={detail}")
            self._speak("That name doesn't match what I just remembered.")
            logging.info(
                "[ForgetLast] id_mismatch requested=%s detail=%s",
                requested,
                detail,
            )
            return

        if err == "no_safe_files":
            self._write_status("result=no_safe_files")
            self._speak("I couldn't find the right files to remove.")
            logging.error("[ForgetLast] no_safe_files")
            return

        if err == "recognition_disabled":
            self._write_status("result=recognition_disabled")
            self._speak("I can't undo that right now.")
            logging.error("[ForgetLast] recognition_disabled")
            return

        # Unknown
        self._write_status(f"result=unknown error={err}")
        self._speak("Something went wrong undoing that.")
        logging.error("[ForgetLast] unknown error: %s", resp)
