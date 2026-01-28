import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

from .singleton import singleton


@dataclass
class IdentitiesSnapshot:
    """
    Canonical record returned by `/gallery/identities`.

    Attributes
    ----------
    ts : float
        UNIX epoch seconds (local fallback).
    total : int
        Total number of identities in the gallery.
    names : list[str]
        Identity labels.
    raw : dict
        Full response body for advanced consumers.

    Methods
    -------
    to_text() -> str
        Produce the compact human-readable line the LLM/fuser expects.
    """

    ts: float
    total: int
    names: List[str]
    raw: Dict

    def to_text(self) -> str:
        """
        Produce a concise one-line summary of the identities snapshot.

        Returns
        -------
        str
            Formatted string like "total=3 ids=[Alice, Bob, Charlie]".
        """
        seen = set()
        ordered = []
        for n in self.names or []:
            n = (n or "").strip()
            if n and n not in seen:
                seen.add(n)
                ordered.append(n)
        return f"total={self.total} ids=[{', '.join(ordered)}]"


@singleton
class GalleryIdentitiesProvider:
    """
    Singleton provider that polls `/gallery/identities` and emits text lines.

    Tasks
    -----
    - Background thread POSTs to `{base_url}/gallery/identities` at a cadence.
    - Converts JSON to the concise string via `IdentitiesSnapshot.to_text()`.
    - Invokes every registered callback with that string.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:6793",
        fps: float = 1.0,
        timeout_s: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        base_url : str
            Base HTTP URL of the face service (e.g., "http://127.0.0.1:6793").
        fps : float
            Polling rate (events/sec). 1.0 → every 1s.
        timeout_s : float
            HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._callbacks: List[Callable[[str], None]] = []
        self._cb_lock = threading.Lock()

        self._session = requests.Session()

    def register_message_callback(self, fn: Callable[[str], None]) -> None:
        """
        Subscribe a consumer to receive each emitted gallery identities line.

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
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="gallery-identities-poll", daemon=True
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """Request the background thread to stop."""
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        """
        Internal polling loop (runs on the provider's background thread).

        Overview
        --------
        Sleeps between polls based on `fps`, fetches the gallery list, converts it
        to a summary line, and emits to registered callbacks (respecting emit policy).
        """
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.02, next_t - now))
                continue
            try:
                snap = self._fetch_snapshot()
                self._emit(snap.to_text())
            except Exception as e:
                logging.warning(
                    f"Failed to fetch/emit gallery identities snapshot: {e}"
                )

            next_t += self.period
            if next_t < time.time() - self.period:
                next_t = time.time()

    def _emit(self, text: str) -> None:
        """
        Invoke all registered callbacks with the given summary line.

        Parameters
        ----------
        text : str
            Preformatted gallery line to deliver.
        """
        with self._cb_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(text)
            except Exception as e:
                logging.warning(f"Gallery identities callback failed: {e}")

    def _fetch_snapshot(self) -> IdentitiesSnapshot:
        """
        Fetch and parse `/gallery/identities` into a structured record.

        Overview
        --------
        Issues a POST with `{}` to `{base_url}/gallery/identities`, validates the
        response, and maps it to a small structure (e.g., total count and name list).

        Returns
        -------
        GalleryIdentityList
            Structured view of the current gallery (e.g., `total`, `names`, raw JSON).
        """
        url = f"{self.base_url}/gallery/identities"
        r = self._session.post(url, json={}, timeout=self.timeout_s)  # type: ignore
        r.raise_for_status()
        data = r.json() or {}

        ok = bool(data.get("ok"))
        if not ok:
            # Graceful empty snapshot on bad response
            return IdentitiesSnapshot(ts=time.time(), total=0, names=[], raw=data)

        total = int(data.get("total", 0) or 0)
        raw_identities = data.get("identities", []) or []
        names = []
        try:
            for item in raw_identities:
                if isinstance(item, dict):
                    n = str(item.get("id", "")).strip()
                    if n:
                        names.append(n)
        except Exception as e:
            logging.warning(f"Failed to parse gallery identities: {e}")
            names = []

        # Use local time; server may not return one
        ts = time.time()
        return IdentitiesSnapshot(ts=ts, total=total, names=names, raw=data)
