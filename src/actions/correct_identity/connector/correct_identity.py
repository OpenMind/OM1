"""
Correct Identity connector for OM1.

Renames a recently-enrolled identity by calling /gallery/move_samples on
the face API. Used when the user disputes the LABEL (not the person) of
a just-enrolled identity.

The connector is stateless — the face API enforces the 60s TTL on
last_enrollment, so this connector can't accidentally rename an old
identity even if the LLM asks it to.
"""

import asyncio
import logging
import re
import time
from typing import Dict, Optional

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.correct_identity.interface import CorrectIdentityInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider

_DEDUP_SUFFIX_RE = re.compile(r"_\d+$")


class CorrectIdentityConfig(ActionConfig):
    """
    Configuration for CorrectIdentity connector.

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


class CorrectIdentityConnector(ActionConnector[CorrectIdentityConfig, CorrectIdentityInput]):
    """Rename a recently-enrolled identity via /gallery/move_samples."""

    def __init__(self, config: CorrectIdentityConfig):
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
        """Queue a TTS message. No-op if message is None or empty."""
        if not message:
            return
        try:
            self.elevenlabs_tts_provider.add_pending_message(message)
        except Exception as e:
            logging.warning("TTS queue failed: %s", e)

    @staticmethod
    def _display_name(id_str: str) -> str:
        """
        Convert internal id to natural display form for TTS.
        'wendy_1' → 'Wendy', 'jerin-peter' → 'Jerin Peter'.
        """
        cleaned = _DEDUP_SUFFIX_RE.sub("", id_str)
        cleaned = cleaned.replace("-", " ").replace("_", " ")
        return cleaned.title()

    # -------- Main entry point --------

    async def connect(self, output_interface: CorrectIdentityInput) -> None:
        """Execute a single identity rename/merge."""
        from_id = (output_interface.from_id or "").strip().lower()
        to_id = (output_interface.to_id or "").strip().lower()

        # Local validation — catch obvious LLM mistakes before HTTP
        if not from_id or not to_id:
            self._write_status(f"result=bad_id from={from_id!r} to={to_id!r}")
            logging.error("[CorrectIdentity] missing ids: from=%r to=%r", from_id, to_id)
            # Silent — LLM-side issue, shouldn't reach user
            return

        if from_id == to_id:
            self._write_status(f"result=same_id id={from_id}")
            logging.info("[CorrectIdentity] no-op: from_id == to_id")
            # Silent — no change needed
            return

        loop = asyncio.get_running_loop()
        body = {"from_id": from_id, "to_id": to_id}
        resp = await loop.run_in_executor(None, self._post_json, "/gallery/move_samples", body)

        self._dispatch_response(resp, from_id, to_id)

    def _dispatch_response(self, resp: Optional[Dict], from_id: str, to_id: str) -> None:
        """Map the API response to status + TTS."""
        if resp is None:
            self._write_status("result=network_error")
            self._speak("I had trouble updating that.")
            logging.error("[CorrectIdentity] network error")
            return

        if resp.get("ok"):
            moved = int(resp.get("moved", 0))
            from_removed = bool(resp.get("from_removed"))
            display = self._display_name(to_id)
            self._write_status(
                f"result=success from={from_id} to={to_id} "
                f"moved={moved} from_removed={'true' if from_removed else 'false'}"
            )
            self._speak(f"Got it, I've updated your name to {display}.")
            logging.info(
                "[CorrectIdentity] ok from=%s to=%s moved=%d",
                from_id,
                to_id,
                moved,
            )
            return

        err = str(resp.get("error", "unknown"))

        if err in ("no_recent_enrollment", "stale_enrollment"):
            self._write_status(f"result={err}")
            self._speak("I can't change that — too much time has passed since I remembered you.")
            logging.info("[CorrectIdentity] %s", err)
            return

        if err == "bad_id":
            detail = str(resp.get("detail", ""))
            self._write_status(f"result=bad_id detail={detail}")
            # Silent — LLM produced invalid id; should re-prompt user
            logging.error("[CorrectIdentity] bad_id: %s", detail)
            return

        if err == "same_id":
            self._write_status(f"result=same_id id={from_id}")
            logging.info("[CorrectIdentity] same_id (caught by API)")
            return

        if err == "no_safe_files":
            self._write_status("result=no_safe_files")
            self._speak("I couldn't find the right files to update.")
            logging.error("[CorrectIdentity] no_safe_files")
            return

        if err == "recognition_disabled":
            self._write_status("result=recognition_disabled")
            self._speak("I can't update names right now.")
            logging.error("[CorrectIdentity] recognition_disabled")
            return

        # Unknown
        self._write_status(f"result=unknown error={err}")
        self._speak("Something went wrong updating that.")
        logging.error("[CorrectIdentity] unknown error: %s", resp)
