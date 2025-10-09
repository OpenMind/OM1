# src/providers/face_presence_provider.py
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # we'll fall back to urllib

from .singleton import singleton


@dataclass(frozen=True)
class PresenceSnapshot:
    """
    Canonical record from /who.

    Attributes
    ----------
    ts : float
        Server timestamp (seconds). Falls back to local time if missing.
    names_now : list[str]
        Known identities currently present (deduped).
    unknown_now : int
        Count of unknown faces currently present.
    raw : dict
        The full JSON body returned by /who.
    """

    ts: float
    names_now: List[str]
    unknown_now: int
    raw: Dict

    def to_text(self) -> str:
        k = ", ".join(self.names_now) if self.names_now else "none"
        return f"present=[{k}], unknown={self.unknown_now}, ts={self.ts:.3f}"


@singleton
class FacePresenceProvider:
    """
    Polls the face stream HTTP API (/who) at a fixed cadence and buffers results.

    Usage
    -----
    provider = FacePresenceProvider(
        base_url="http://127.0.0.1:6793", recent_sec=2.0, fps=5.0, capacity=300
    )
    provider.start()
    snap = provider.peek_latest()
    hist = provider.get_history_since(time.time() - 5)
    provider.stop()

    Thread-safety
    -------------
    All buffer operations are guarded by an internal lock.
    """

    def __init__(
        self,
        *,
        base_url: str,
        recent_sec: float = 2.0,
        fps: float = 5.0,
        timeout_s: float = 2.0,
        capacity: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.recent_sec = float(recent_sec)
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)

        # Ring buffer that drops oldest when full.
        self._buf: Deque[PresenceSnapshot] = deque(maxlen=int(capacity))
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------- lifecycle ---------------- #

    def start(self) -> None:
        """Start background polling thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="face-presence-poll", daemon=True
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """Signal the thread to stop; optionally join."""
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3.0)

    # ---------------- consumption API ---------------- #

    def drain_latest(self) -> Optional[PresenceSnapshot]:
        """
        Pop and return the newest snapshot; clear older ones.

        Returns
        -------
        PresenceSnapshot | None
        """
        with self._lock:
            if not self._buf:
                return None
            latest = self._buf[-1]
            self._buf.clear()
            self._buf.append(latest)
            return latest

    def peek_latest(self) -> Optional[PresenceSnapshot]:
        """
        Return newest snapshot without clearing the buffer.

        Returns
        -------
        PresenceSnapshot | None
        """
        with self._lock:
            return self._buf[-1] if self._buf else None

    def get_history_since(self, since_ts: float) -> List[PresenceSnapshot]:
        """
        Return snapshots with ts >= since_ts (ascending).

        Note: History length is bounded by `capacity`.

        Parameters
        ----------
        since_ts : float

        Returns
        -------
        list[PresenceSnapshot]
        """
        with self._lock:
            return [it for it in self._buf if it.ts >= since_ts]

    # ---------------- internals ---------------- #

    def _loop(self) -> None:
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.02, next_t - now))
                continue

            try:
                snap = self._fetch_snapshot()
                with self._lock:
                    self._buf.append(snap)  # drops oldest when full
            except Exception:
                # swallow transient errors; keep polling
                pass

            next_t += self.period
            # if we fell behind (e.g., network stall), catch up sanely
            late = time.time() - next_t
            if late > self.period:
                next_t = time.time()

    def _fetch_snapshot(self) -> PresenceSnapshot:
        """POST /who {recent_sec} and map to PresenceSnapshot."""
        body = {"recent_sec": self.recent_sec}
        url = f"{self.base_url}/who"

        if requests is None:
            from urllib.request import Request, urlopen

            req = Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=self.timeout_s) as r:  # nosec B310 (internal URL)
                data = json.loads(r.read().decode("utf-8"))
        else:
            r = requests.post(url, json=body, timeout=self.timeout_s)  # type: ignore
            r.raise_for_status()
            data = r.json()

        names = list(data.get("now", []) or [])
        unknown = int(data.get("unknown_now", 0) or 0)
        ts = float(data.get("server_ts", time.time()))
        return PresenceSnapshot(ts=ts, names_now=names, unknown_now=unknown, raw=data)
