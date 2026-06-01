"""
Selfie connector for OM1.

Calls the multi-frame /selfie endpoint exposed by om1_video_processor and
translates each response code into:
  - a SelfieStatus key-value line for the LLM (via io_provider)
  - a brief neutral TTS confirmation for the user (via elevenlabs)
  - light connector-side state (last_enrolled_id, last_match_name) as
    debugging breadcrumbs

The connector does NOT own conversational persona or correction logic.
Persona is the LLM's job on the next turn after reading SelfieStatus.
Correction decisions (correct_identity / forget_last) are separate actions
governed by the API's own 60s TTL.
"""

import asyncio
import logging
import re
import time
from typing import Dict, Optional

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.selfie.interface import SelfieInput
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider

# Strip dedup-system suffix like "wendy_1" → "wendy"
_DEDUP_SUFFIX_RE = re.compile(r"_\d+$")


class SelfieConfig(ActionConfig):
    """
    Configuration for Selfie connector.

    Parameters
    ----------
    face_http_base_url : str
        Base URL for the face HTTP service.
    face_recent_sec : float
        Recency window in seconds for the /who pre-check.
    poll_ms : int
        Polling interval in milliseconds for the pre-check.
    timeout_sec : int
        Max seconds to wait for at least one face to appear before giving up.
    http_timeout_sec : float
        HTTP request timeout in seconds. Must exceed the /selfie collection
        window (~1.5s) with margin.
    """

    face_http_base_url: str = Field(
        default="http://127.0.0.1:6793",
        description="Base URL for the face HTTP service.",
    )
    face_recent_sec: float = Field(
        default=1.0,
        description="Recency window in seconds for the /who pre-check.",
    )
    poll_ms: int = Field(
        default=200,
        description="Polling interval in milliseconds for the pre-check.",
    )
    timeout_sec: int = Field(
        default=8,
        description="Max seconds to wait for ≥1 face before giving up.",
    )
    http_timeout_sec: float = Field(
        default=5.0,
        description="HTTP request timeout in seconds.",
    )


class SelfieConnector(ActionConnector[SelfieConfig, SelfieInput]):
    """
    Enroll a face through the multi-frame /selfie endpoint.

    Each API response code maps to one row of the outcome table; see
    `_dispatch_response` for the full set. The connector only handles
    enrollment — corrections are separate actions.
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

        self.elevenlabs_tts_provider = ElevenLabsTTSProvider()
        self.io_provider = IOProvider()

        # Logging breadcrumbs for the most recent attempt.
        # The API server is the source of truth for the 60s correction TTL —
        # these are not enforced here.
        self.last_enrolled_id: Optional[str] = None
        self.last_match_name: Optional[str] = None

    def _post_json(self, path: str, body: Dict) -> Optional[Dict]:
        """
        POST JSON to the face service.

        Parameters
        ----------
        path : str
            Endpoint path (e.g., "/who", "/selfie").
        body : Dict
            Request body.

        Returns
        -------
        Optional[Dict]
            Parsed JSON dict on success; None on transport error.
        """
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=body, timeout=self.http_timeout)
            return r.json()
        except Exception as e:
            logging.warning("HTTP POST %s failed (%s) body=%s", url, e, body)
            return None

    def _get_config(self) -> Dict:
        """Fetch current service config. Returns {} on error."""
        resp = self._post_json("/config", {"get": True}) or {}
        return resp if isinstance(resp, dict) else {}

    def _set_blur(self, on: bool) -> None:
        """Enable/disable blur on the service (affects RTSP viewing)."""
        _ = self._post_json("/config", {"set": {"blur": bool(on)}})

    def _who_snapshot(self) -> Optional[Dict]:
        """Query current faces within the recency window."""
        return self._post_json("/who", {"recent_sec": self.recent_sec})

    def _wait_any_face(self, timeout_sec: int) -> bool:
        """
        Poll /who until at least one face is visible, or timeout.

        Multi-person ambiguity is handled by the /selfie API itself — this
        pre-check only ensures we're not calling /selfie into an empty scene.

        Parameters
        ----------
        timeout_sec : int
            Maximum seconds to wait. <=0 uses default_timeout.

        Returns
        -------
        bool
            True if at least one face appeared within the timeout.
        """
        if timeout_sec <= 0:
            timeout_sec = self.default_timeout
        tries = max(1, int((timeout_sec * 1000) / self.poll_ms))
        for _ in range(tries):
            resp = self._who_snapshot() or {}
            now = resp.get("now") or []
            unknown_now = int(resp.get("unknown_now") or 0)
            if len(now) + unknown_now >= 1:
                logging.info(
                    "[Selfie] pre-check ok (known=%s, unknown=%d)",
                    now,
                    unknown_now,
                )
                return True
            self.sleep(self.poll_ms / 1000.0)
        logging.info("[Selfie] pre-check: no face appeared within %ds", timeout_sec)
        return False

    def _write_status(self, line: str) -> None:
        """Surface the result to the LLM as a SelfieStatus input line."""
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
        Convert internal id to a natural display form for TTS.

        - 'wendy_1'      → 'Wendy'        (strip dedup suffix)
        - 'jerin-peter'  → 'Jerin Peter'  (dashes back to spaces, title case)
        - 'wendy'        → 'Wendy'

        Underscores within the name (not the trailing _N suffix) also become
        spaces, since both '-' and '_' are valid in-name separators per the
        API's name policy.
        """
        cleaned = _DEDUP_SUFFIX_RE.sub("", id_str)
        cleaned = cleaned.replace("-", " ").replace("_", " ")
        return cleaned.title()

    def _clear_state(self) -> None:
        """Clear breadcrumb state (called on any failure)."""
        self.last_enrolled_id = None
        self.last_match_name = None

    def _dispatch_response(self, resp: Dict, claimed_id: str) -> None:
        """
        Map the /selfie API response to SelfieStatus + TTS + state.

        Parameters
        ----------
        resp : Dict
            Parsed API response.
        claimed_id : str
            The id the user said they were (before any API renaming).
        """
        # ---- Success path ----
        if resp.get("ok"):
            saved_id = str(resp.get("id", claimed_id))
            merged = bool(resp.get("merged"))
            samples = int(resp.get("samples_saved", 0))
            display = self._display_name(saved_id)

            self.last_enrolled_id = saved_id
            self.last_match_name = None

            tag = "merged" if merged else "success"
            self._write_status(
                f"result={tag} id={saved_id} samples={samples} " f"merged={'true' if merged else 'false'}"
            )
            if merged:
                self._speak(f"Welcome back, {display}!")
            else:
                self._speak(f"Nice to meet you, {display}! I'll remember you next time.")
            logging.info("[Selfie] %s id=%s samples=%d", tag, saved_id, samples)
            return

        err = str(resp.get("error", "unknown"))

        if err == "ambiguous_subjects":
            n = int(resp.get("n_engaged", 0))
            self._write_status(f"result=ambiguous engaged={n}")
            self._speak("I see a few people. Could you step closer so I can focus on you?")
            logging.info("[Selfie] ambiguous engaged=%d", n)
            self._clear_state()
            return

        if err == "face_belongs_to":
            matched = str(resp.get("name", "someone"))
            sim = float(resp.get("sim", 0.0))
            matched_display = self._display_name(matched)
            self.last_match_name = matched
            self._write_status(f"result=face_belongs_to claimed={claimed_id} " f"matched={matched} sim={sim:.3f}")
            self._speak(f"You look a lot like {matched_display}. " f"Are you {matched_display}, or someone different?")
            logging.info(
                "[Selfie] face_belongs_to claimed=%s matched=%s sim=%.3f",
                claimed_id,
                matched,
                sim,
            )
            return

        if err == "no_valid_frames":
            self._write_status("result=low_quality")
            self._speak("I can't see your face clearly. Could you look at me directly?")
            logging.info("[Selfie] no_valid_frames")
            self._clear_state()
            return

        if err == "insufficient_samples":
            got = int(resp.get("got", 0))
            self._write_status(f"result=partial got={got}")
            self._speak("Hold still — almost got it.")
            logging.info("[Selfie] insufficient_samples got=%d", got)
            self._clear_state()
            return

        if err == "busy":
            # Only reaches here AFTER the retry in connect()
            self._write_status("result=busy retries=1")
            self._speak("One sec, finishing the last one.")
            logging.info("[Selfie] still busy after retry")
            self._clear_state()
            return

        if err == "bad_id":
            detail = str(resp.get("detail", ""))
            self._write_status(f"result=bad_id detail={detail}")
            # No TTS — bad_id means the LLM produced an invalid id;
            # surface to LLM only so it can re-prompt the user.
            logging.error("[Selfie] bad_id: %s", detail)
            self._clear_state()
            return

        if err == "recognition_disabled":
            self._write_status("result=recognition_disabled")
            self._speak("I can't see right now — please try again in a moment.")
            logging.error("[Selfie] recognition_disabled (gallery manager not loaded)")
            self._clear_state()
            return

        self._write_status(f"result=unknown error={err}")
        self._speak("Something went wrong. Could you try again?")
        logging.error("[Selfie] unknown error: %s", resp)
        self._clear_state()

    def _dispatch_network_error(self) -> None:
        """Called when /selfie returns None (HTTP transport failure)."""
        self._write_status("result=network_error")
        self._speak("I lost connection for a moment.")
        logging.error("[Selfie] network error talking to face API")
        self._clear_state()

    async def connect(self, output_interface: SelfieInput) -> None:
        """
        Execute a single selfie enrollment attempt.

        Reads from `output_interface`:
          - action       : str  — claimed id, e.g. "wendy" (required)
          - timeout_sec  : int  — max wait for a face to appear (default 8)
          - force        : bool — bypass cross-name reject; default False

        Writes SelfieStatus and queues TTS per the outcome table.
        """
        name = (output_interface.action or "").strip()
        timeout_sec = int(output_interface.timeout_sec or self.default_timeout)
        force = bool(getattr(output_interface, "force", False))

        if not name:
            logging.error("[Selfie] empty id; expected something like 'wendy'")
            self._write_status("result=bad_id detail=empty")
            # No TTS — LLM should have produced a valid id
            return

        loop = asyncio.get_running_loop()

        # Snapshot blur state, disable for the enrollment so the demo RTSP
        # view shows the actual face being captured.
        cfg = await loop.run_in_executor(None, self._get_config)
        orig_blur = bool(((cfg or {}).get("config") or {}).get("blur", True))
        await loop.run_in_executor(None, self._set_blur, False)

        try:
            # Pre-check: fast-fail if nobody is in frame.
            # (The API would eventually return no_valid_frames after 1.5s,
            # but a pre-check gives quicker feedback.)
            face_present = await loop.run_in_executor(None, self._wait_any_face, timeout_sec)
            if not face_present:
                self._write_status("result=low_quality reason=no_one_present")
                self._speak("I don't see anyone in front of me yet.")
                return

            # Call /selfie — the API runs its 1.5s multi-frame collection
            body = {"id": name, "force": force}
            resp = await loop.run_in_executor(None, self._post_json, "/selfie", body)

            # Retry once on transient busy
            if isinstance(resp, dict) and resp.get("error") == "busy":
                logging.info("[Selfie] busy on first call, retrying in 1s")
                await asyncio.sleep(1.0)
                resp = await loop.run_in_executor(None, self._post_json, "/selfie", body)

            # Dispatch
            if resp is None:
                self._dispatch_network_error()
            else:
                self._dispatch_response(resp, claimed_id=name)

        finally:
            # Always restore original blur state
            await loop.run_in_executor(None, self._set_blur, orig_blur)
