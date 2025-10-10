import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

from .singleton import singleton


@dataclass
class PresenceSnapshot:
    """
    Canonical record returned by `/who`.

    Attributes
    ----------
    ts : float
        Server timestamp in UNIX epoch seconds (falls back to local time if missing).
    names_now : list[str]
        Known identities currently present (deduplicated).
    unknown_now : int
        Count of unknown faces currently present.
    raw : dict
        Full response body from `/who` for advanced consumers.

    Methods
    -------
    to_text() -> str
        Produce a concise human-readable summary suitable for logs/prompts.
    """

    ts: float
    names_now: List[str]
    unknown_now: int
    raw: Dict

    def to_text(self) -> str:
        known = ", ".join(self.names_now) if self.names_now else "none"
        return f"present=[{known}], unknown={self.unknown_now}, ts={self.ts:.3f}"


@singleton
class FacePresenceProvider:
    """
    Singleton provider that polls `/who` at a fixed cadence and emits text lines.

    Tasks
    ------------
    - Spawns one background thread that periodically POSTs to `{base_url}/who`.
    - Converts each JSON snapshot to a concise string via `PresenceSnapshot.to_text()`.
    - Invokes every registered callback with that string (same polling thread).
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:6793",
        recent_sec: float = 2.0,
        fps: float = 5.0,
        timeout_s: float = 2.0,
    ) -> None:
        """
        Configure the provider (first construction establishes the singleton).

        Parameters
        ----------
        base_url : str
            Base HTTP URL of the face stream API (e.g., "http://127.0.0.1:6793").
            The provider will call POST `{base_url}/who`.
        recent_sec : float, default 2.0
            Lookback window passed to `/who` (seconds of presence history).
        fps : float, default 5.0
            Polling rate in events per second (e.g., 5.0 → every 0.2s).
        timeout_s : float, default 2.0
            HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.recent_sec = float(recent_sec)
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._callbacks: List[Callable[[str], None]] = []
        self._cb_lock = threading.Lock()

    def register_message_callback(self, fn: Callable[[str], None]) -> None:
        """
        Subscribe a consumer to receive each emitted presence line.

        Parameters
        ----------
        fn : Callable[[str], None]
            Function invoked from the polling thread with one formatted string.
        """
        with self._cb_lock:
            if fn not in self._callbacks:
                self._callbacks.append(fn)

    def unregister_message_callback(self, fn: Callable[[str], None]) -> None:
        """
        Remove a previously registered consumer.

        Parameters
        ----------
        fn : Callable[[str], None]
            The same callable passed to `register_message_callback()`.
        """
        with self._cb_lock:
            try:
                self._callbacks.remove(fn)
            except ValueError:
                pass

    def start(self) -> None:
        """Start the background polling thread"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="face-presence-poll", daemon=True
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """Request the background thread to strop"""
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        """
        Internal polling loop.

        Tasks
        --------
        - Waits until the next scheduled time (based on `fps`).
        - Calls `_fetch_snapshot()` → formats with `to_text()` → `_emit(text)`.
        """
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.02, next_t - now))
                continue
            try:
                snap = self._fetch_snapshot()
                text = snap.to_text()
                self._emit(text)
            except Exception:
                pass

            next_t += self.period
            if next_t < time.time() - self.period:
                next_t = time.time()

    def _emit(self, text: str) -> None:
        """
        Deliver one formatted presence line to all subscribers.

        Parameters
        ----------
        text : str
            A concise, human-readable snapshot (e.g., "present=[alice], unknown=0, ts=...").
        """
        with self._cb_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(text)
            except Exception:
                pass

    def _fetch_snapshot(self) -> PresenceSnapshot:
        """
        POST `/who` and map the response to a `PresenceSnapshot`.

        Returns
        -------
        PresenceSnapshot
            Structured view of the server's `/who` response.
        """
        body = {"recent_sec": self.recent_sec}
        url = f"{self.base_url}/who"

        r = requests.post(url, json=body, timeout=self.timeout_s)  # type: ignore
        r.raise_for_status()
        data = r.json()

        names = list(data.get("now", []) or [])
        unknown = int(data.get("unknown_now", 0) or 0)
        ts = float(data.get("server_ts", time.time()))
        return PresenceSnapshot(ts=ts, names_now=names, unknown_now=unknown, raw=data)
