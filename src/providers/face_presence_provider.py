import logging
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
    names : list[str]
        Known identities present (deduplicated).
    unknown_faces : int
        Count of unknown faces present.
    raw : dict
        Full response body from `/who` for advanced consumers.

    Methods
    -------
    to_text() -> str
        Produce a concise human-readable summary suitable for logs/prompts.
    """

    ts: float
    names: List[str]
    unknown_faces: int
    raw: Dict

    def to_text(self) -> str:
        """
        Produce a concise, natural sentence without timestamps, handling
        any number of known people and unknown faces.

        Examples
        --------
        - names=["wendy"], unknown=0
        -> "In Camera View: 1 known (wendy)."
        - names=["wendy","alice","bob"], unknown=2
        -> "In Camera view: 3 known (wendy, alice and bob) and 2 unknown faces."
        - names=[], unknown=1
        -> "In Camera view: 1 unknown face."
        - names=[], unknown=0
        -> "No one in view."
        """
        seen = set()
        clean: List[str] = []
        for n in self.names or []:
            n = (n or "").strip()
            if n and n.lower() != "unknown" and n not in seen:
                seen.add(n)
                clean.append(n)

        k = len(clean)
        u = int(self.unknown_faces or 0)

        def join_names(ns: List[str]) -> str:
            if not ns:
                return ""
            if len(ns) == 1:
                return ns[0]
            if len(ns) == 2:
                return f"{ns[0]} and {ns[1]}"
            return ", ".join(ns[:-1]) + f" and {ns[-1]}"

        if k == 0 and u == 0:
            return "No one in view."

        parts = []
        if k > 0:
            parts.append(f"{k} known ({join_names(clean)})")
        if u > 0:
            parts.append(f"{u} unknown face" + ("s" if u != 1 else ""))

        return "In Camera View: " + " and ".join(parts) + "."


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
        recent_sec: float = 3.0,
        fps: float = 5.0,
        timeout_s: float = 2.0,
        prefer_recent: bool = True,
        unknown_frac_threshold: float = 0.15,
        unknown_min_count: int = 6,
        min_obs_window: int = 24,
    ) -> None:
        """
        Configure the provider (first construction establishes the singleton).

        Parameters
        ----------
        base_url : str, optional
            Base HTTP URL of the face stream API (e.g., "http://127.0.0.1:6793").
            The provider will call POST `{base_url}/who`. Defaults to "http://127.0.0.1:6793".
        recent_sec : float, optional
            Lookback window passed to `/who` (seconds of presence history).
            Defaults to 3.0.
        fps : float, optional
            Polling rate in events per second (e.g., 5.0 → every 0.2s).
            Defaults to 5.0.
        timeout_s : float, optional
            HTTP request timeout in seconds. Defaults to 2.0.
        prefer_recent : bool, optional
            If True, prioritize recent face detection data when fetching snapshots.
            Defaults to True.
        unknown_frac_threshold : float, optional
            Fraction threshold for suppressing unknown face counts based on recent frames.
            Defaults to 0.15.
        unknown_min_count : int, optional
            Minimum count of unknown faces required before applying suppression logic.
            Defaults to 6.
        min_obs_window : int, optional
            Minimum observation window size (in frames) used for unknown face suppression.
            Defaults to 24.
        """
        self.base_url = base_url.rstrip("/")
        self.recent_sec = float(recent_sec)
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)
        self.prefer_recent = bool(prefer_recent)
        self.unknown_frac_threshold = float(unknown_frac_threshold)
        self.unknown_min_count = int(unknown_min_count)
        self.min_obs_window = int(min_obs_window)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List = []
        self._cb_lock = threading.Lock()
        self._session = requests.Session()
        self._unknown_faces: int = 0

    def set_recent_sec(self, sec: float) -> None:
        """Dynamically change the lookback window used for `/who`."""
        self.recent_sec = max(0.0, float(sec))

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
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="face-presence-poll", daemon=True
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """Request the background thread to stop."""
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
            except Exception as e:
                logging.warning(f"Failed to fetch/emit face presence snapshot: {e}")

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
            except Exception as e:
                logging.warning(f"Face presence callback failed: {e}")

    def _fetch_snapshot(self, recent_sec: Optional[float] = None) -> PresenceSnapshot:
        """
        POST `/who` with a lookback window (default: self.recent_sec) and build a
        turn-friendly snapshot using **frames-based** suppression for unknowns.

        Suppression rule:
          If (frames_recent >= min_obs_window) AND
             (frames_with_unknown / frames_recent < unknown_frac_threshold),
          then suppress unknowns (report 0). Otherwise, report the **peak**
          number of unknown faces observed in any single frame within the window.

        Parameters
        ----------
        recent_sec : Optional[float]
            Lookback window in seconds (overrides self.recent_sec if given).

        Returns
        -------
        PresenceSnapshot
            The canonical presence snapshot.
        """
        sec = float(self.recent_sec if recent_sec is None else recent_sec)
        url = f"{self.base_url}/who"
        r = self._session.post(url, json={"recent_sec": sec}, timeout=self.timeout_s)
        r.raise_for_status()
        data: Dict = r.json() or {}

        if self.prefer_recent:

            name_frames: Dict[str, int] = data.get("recent_name_frames", {}) or {}
            names = [k for k in name_frames.keys() if k and k != "unknown"]

            # Frames-based unknown suppression
            frames_recent = int(data.get("frames_recent", 0) or 0)
            frames_with_unknown = int(data.get("frames_with_unknown", 0) or 0)
            unknown_peak = int(data.get("unknown_recent", 0) or 0)

            if frames_recent > 0:
                unknown_frac = frames_with_unknown / float(frames_recent)
                if (
                    frames_recent >= self.min_obs_window
                    and unknown_frac < self.unknown_frac_threshold
                ):
                    unknown_faces = 0  # suppress brief/rare unknowns
                else:
                    unknown_faces = unknown_peak  # report the maximum unknown seen in any single frame
            else:
                now = data.get("now", []) or []
                seen, names_fallback = set(), []
                for n in now:
                    if n and n != "unknown" and n not in seen:
                        seen.add(n)
                        names_fallback.append(n)
                names = names_fallback
                unknown_faces = int(data.get("unknown_now", 0) or 0)
        else:
            now = data.get("now", []) or []
            seen, names = set(), []
            for n in now:
                if n and n != "unknown" and n not in seen:
                    seen.add(n)
                    names.append(n)
            unknown_faces = int(data.get("unknown_now", 0) or 0)

        ts = float(data.get("server_ts", time.time()))

        self._unknown_faces = int(unknown_faces)

        return PresenceSnapshot(
            ts=ts, names=names, unknown_faces=unknown_faces, raw=data
        )

    @property
    def unknown_faces(self) -> int:
        """Most recent (suppressed) count of unknown faces detected in the lookback window."""
        return self._unknown_faces
