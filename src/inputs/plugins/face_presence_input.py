# src/inputs/plugins/face_presence_input.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

from providers.face_presence_provider import FacePresenceProvider, PresenceSnapshot


@dataclass(frozen=True)
class FacePresenceReading:
    """
    Lightweight value object for downstream consumers / LLM prompts.

    Attributes
    ----------
    ts : float
        Timestamp of the snapshot.
    names_now : list[str]
        Known identities present.
    unknown_now : int
        Count of unknown faces present.
    text : str
        Human-friendly summary, suitable for prompt injection.
    raw : dict
        Full body from the server (optional use).
    """
    ts: float
    names_now: List[str]
    unknown_now: int
    text: str
    raw: dict


class FacePresenceInput:
    """
    Async input facade (mirrors the style of dimo_tesla / vlm_local_yolo):

      - start()/stop(): manage an internal polling coroutine that samples the provider
        at a fixed interval (default 0.2s) and caches the latest reading.

      - get_latest(): return newest item and clear older provider entries.
      - peek(): non-destructive read of the newest item.
      - formatted_latest_buffer(): compact string (or multi-line) for LLM prompts.

    This class does *not* talk to HTTP directly; it consumes the provider’s buffer.
    """

    def __init__(self, provider: FacePresenceProvider, *, poll_interval_s: float = 0.2) -> None:
        self._provider = provider
        self._poll_interval = float(poll_interval_s)
        self._task: Optional[asyncio.Task] = None
        self._last_ts: float = 0.0
        self._latest: Optional[FacePresenceReading] = None
        self._lock = asyncio.Lock()

    # --------------- lifecycle --------------- #

    async def start(self) -> None:
        """Ensure provider is running and start the async poller."""
        self._provider.start()
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._poll_loop(), name="face-presence-input-poll")

    async def stop(self) -> None:
        """Cancel the poll task (provider can be shared; we don't stop it here)."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # --------------- public API --------------- #

    def get_latest(self) -> Optional[FacePresenceReading]:
        """
        Pop newest snapshot from provider (clears older ones) and map to reading.

        Returns
        -------
        FacePresenceReading | None
        """
        snap = self._provider.drain_latest()
        if not snap:
            return None
        return self._raw_to_input(snap)

    def peek(self) -> Optional[FacePresenceReading]:
        """
        Non-destructive read of newest snapshot.

        Returns
        -------
        FacePresenceReading | None
        """
        snap = self._provider.peek_latest()
        if not snap:
            return None
        return self._raw_to_input(snap)

    def formatted_latest_buffer(self, history_sec: float = 2.0) -> str:
        """
        Return a compact, human-friendly summary for LLM prompts.

        Parameters
        ----------
        history_sec : float
            How much history (seconds) to summarize. Bound by provider capacity.

        Returns
        -------
        str
        """
        since = time.time() - float(history_sec)
        hist = self._provider.get_history_since(since)
        if not hist:
            return "face_presence: no recent detections"

        # Keep it concise: last 3 lines
        lines = [self._raw_to_text(self._raw_to_input(s)) for s in hist[-1:]]
        return "face_presence:\n" + "\n".join(f"- {ln}" for ln in lines)

    # --------------- internals --------------- #

    async def _poll_loop(self) -> None:
        """
        Periodically sample provider.peek_latest() and cache the newest reading.

        This mirrors the pattern where inputs maintain a small, always-fresh snapshot
        that other subsystems can read synchronously without awaiting HTTP.
        """
        try:
            while True:
                snap = self._provider.peek_latest()
                if snap and snap.ts != self._last_ts:
                    reading = self._raw_to_input(snap)
                    async with self._lock:
                        self._latest = reading
                        self._last_ts = snap.ts
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise

    def _raw_to_input(self, snap: PresenceSnapshot) -> FacePresenceReading:
        """Map provider snapshot → input reading."""
        return FacePresenceReading(
            ts=snap.ts,
            names_now=snap.names_now,
            unknown_now=snap.unknown_now,
            text=snap.to_text(),
            raw=snap.raw,
        )

    def _raw_to_text(self, reading: FacePresenceReading) -> str:
        """Format a one-liner for prompts or logs."""
        return reading.text

