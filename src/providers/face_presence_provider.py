# providers/face_presence_provider.py
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import requests  # type: ignore


from common.latest_buffer import LatestBuffer


@dataclass(frozen=True)
class PresenceSnapshot:
    """
    Structured snapshot of 'who is present now' from /who.
    """
    ts: float
    names_now: List[str]
    unknown_now: int
    raw: Dict  # full JSON for advanced consumers

    def to_text(self) -> str:
        known = ", ".join(self.names_now) if self.names_now else "none"
        return f"present: [{known}], unknown={self.unknown_now} @ {self.ts:.3f}"


class FacePresenceProvider:
    """
    Singleton-like provider that polls /who at a fixed rate.

    - Publishes the newest PresenceSnapshot to a LatestBuffer (self.buffer).
    - Optionally keeps a bounded history ring (deque(maxlen=history_maxlen)).

    Usage Example:
        prov = FacePresenceProvider.instance(
            base_url="http://127.0.0.1:6793", recent_sec=2.0, fps=5.0
        )
        prov.start()

        # Read latest (and clear):
        latest = prov.buffer.drain_latest()

        # Or peek without clearing:
        latest2 = prov.buffer.peek_latest()

        # History:
        last_50 = prov.get_history(50)
        since_t = prov.get_history_since(time.time() - 10.0)
    """

    _instance: Optional["FacePresenceProvider"] = None
    _inst_lock = threading.Lock()

    @classmethod
    def instance(cls, *args, **kwargs) -> "FacePresenceProvider":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
            return cls._instance

    def __init__(
        self,
        base_url: str,
        recent_sec: float = 2.0,
        fps: float = 5.0,
        timeout_s: float = 2.0,
        fetch_fn: Optional[Callable[[], Dict]] = None,  
        *,
        keep_history: bool = True,
        history_maxlen: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.recent_sec = float(recent_sec)
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)

        # Latest-only buffer
        self.buffer: LatestBuffer[PresenceSnapshot] = LatestBuffer()

        # Bounded history (optional)
        self._keep_history = bool(keep_history)
        self._history: Optional[Deque[PresenceSnapshot]] = (
            deque(maxlen=int(history_maxlen)) if self._keep_history else None
        )
        self._hist_lock = threading.Lock()

        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-presence")
        self._thread: Optional[threading.Thread] = None

        # For testing or custom transports
        self._fetch_fn = fetch_fn

    # ---------------- lifecycle ---------------- #

    def start(self) -> None:
        """Start background polling thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = False) -> None:
        """Stop background polling."""
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------------- polling loop ---------------- #

    def _loop(self) -> None:
        """
        background scheduler that runs in its own thread and make sure we poll /who
        at a steady rate (fps)
        """
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue

            # single-threaded executor ensures no overlapping polls
            fut = self._executor.submit(self._poll_once_safely)
            try:
                fut.result(timeout=self.timeout_s + 0.5)
            except Exception:
                pass

            next_t += self.period
            # if we drifted too far, catch up but don't spin
            if next_t < time.time() - self.period:
                next_t = time.time()

    def _poll_once_safely(self) -> None:
        """Fetch one snapshot and publish it; never raise."""
        try:
            snap = self._fetch_snapshot()
            # Publish newest
            self.buffer.push(snap)
            # Append history (bounded)
            if self._history is not None:
                with self._hist_lock:
                    self._history.append(snap)
        except Exception:
            # swallow errors; keep ticking
            pass

    def _fetch_snapshot(self) -> PresenceSnapshot:
        """Call /who and adapt to PresenceSnapshot."""
        if self._fetch_fn is not None:
            data = self._fetch_fn()
        else:
            body = {"recent_sec": self.recent_sec}
            url = f"{self.base_url}/who"
            if requests is None:
                # fallback without 'requests'
                from urllib.request import Request, urlopen
                req = Request(url, data=json.dumps(body).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=self.timeout_s) as r:  # nosec B310 (internal URL)
                    data = json.loads(r.read().decode("utf-8"))
            else:
                r = requests.post(url, json=body, timeout=self.timeout_s)  # type: ignore
                r.raise_for_status()
                data = r.json()

        names_now = list(data.get("now", []) or [])
        unknown_now = int(data.get("unknown_now", 0) or 0)
        ts = float(data.get("server_ts", time.time()))
        return PresenceSnapshot(ts=ts, names_now=names_now, unknown_now=unknown_now, raw=data)

    # ---------------- history API ---------------- #

    def get_history(self, n: Optional[int] = None) -> List[PresenceSnapshot]:
        """
        Return last n snapshots (or all in the ring if n is None).
        If history disabled, returns [].
        """
        if self._history is None:
            return []
        with self._hist_lock:
            if n is None:
                return list(self._history)
            n = int(max(0, n))
            return list(self._history)[-n:]

    def get_history_since(self, ts: float) -> List[PresenceSnapshot]:
        """
        Return snapshots with timestamp > ts (bounded by the ring).
        If history disabled, returns [].
        """
        if self._history is None:
            return []
        with self._hist_lock:
            return [s for s in self._history if s.ts > ts]
