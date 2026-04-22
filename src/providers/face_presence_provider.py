import asyncio
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
    timestamp : float
        Server timestamp in UNIX epoch seconds (falls back to local time if missing).
    names : list[str]
        Known identities present (deduplicated).
    unknown_faces : int
        Count of unknown faces present.
    closest_name : str
        Name of the person with the largest face area (closest to camera), or "unknown" if unavailable.
    raw : dict
        Full response body from `/who` for advanced consumers.

    Methods
    -------
    to_text() -> str
        Produce a concise human-readable summary suitable for logs/prompts.
    """

    timestamp: float
    names: List[str]
    unknown_faces: int
    closest_name: str
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
        seen_names = set()
        valid_names: List[str] = []
        for name in self.names or []:
            cleaned_name = (name or "").strip()
            if cleaned_name and cleaned_name.lower() != "unknown" and cleaned_name not in seen_names:
                seen_names.add(cleaned_name)
                valid_names.append(cleaned_name)

        known_count = len(valid_names)
        unknown_count = int(self.unknown_faces or 0)

        def join_names(name_list: List[str]) -> str:
            """
            Join a list of names into a human-readable string.

            Parameters
            ----------
            name_list : List[str]
                List of name strings to join.

            Returns
            -------
            str
                Human-readable string with proper conjunctions.
            """
            if not name_list:
                return ""
            if len(name_list) == 1:
                return name_list[0]
            if len(name_list) == 2:
                return f"{name_list[0]} and {name_list[1]}"
            return ", ".join(name_list[:-1]) + f" and {name_list[-1]}"

        if known_count == 0 and unknown_count == 0:
            return "No one in view."

        parts = []
        if known_count > 0:
            parts.append(f"{known_count} known ({join_names(valid_names)})")
        if unknown_count > 0:
            parts.append(f"{unknown_count} unknown face" + ("s" if unknown_count != 1 else ""))

        result = "In Camera View: " + " and ".join(parts) + "."

        if self.closest_name:
            result += f" Closest: {self.closest_name}."

        return result


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
        recent_sec: float = 1.0,
        fps: float = 5.0,
        timeout_s: float = 2.0,
        unknown_frac_threshold: float = 0.15,
        unknown_min_count: int = 6,
        min_obs_window: int = 24,
        min_face_area: int = 500,
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
        unknown_frac_threshold : float, optional
            Fraction threshold for suppressing unknown face counts based on recent frames.
            Defaults to 0.15.
        unknown_min_count : int, optional
            Minimum count of unknown faces required before applying suppression logic.
            Defaults to 6.
        min_obs_window : int, optional
            Minimum observation window size (in frames) used for unknown face suppression.
            Defaults to 24.
        min_face_area : int, optional
            Minimum area (in pixels) for detected faces to be included in the snapshot.
            Defaults to 500.
        """
        self.base_url = base_url.rstrip("/")
        self.recent_sec = float(recent_sec)
        self.period = 1.0 / max(1e-6, float(fps))
        self.timeout_s = float(timeout_s)
        self.unknown_frac_threshold = float(unknown_frac_threshold)
        self.unknown_min_count = int(unknown_min_count)
        self.min_obs_window = int(min_obs_window)
        self.min_face_area = int(min_face_area)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List = []
        self._callback_lock = threading.Lock()
        self._session = requests.Session()
        self._unknown_faces: int = 0

    def set_recent_sec(self, sec: float) -> None:
        """
        Dynamically change the lookback window used for `/who`.

        Parameters
        ----------
        sec : float
            The number of seconds to use as the lookback window.
        """
        self.recent_sec = max(0.0, float(sec))

    def register_message_callback(self, callback: Callable[[str], None]) -> None:
        """
        Subscribe a consumer to receive each emitted presence line.

        Parameters
        ----------
        callback : Callable[[str], None]
            Function invoked from the polling thread with one formatted string.
        """
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_message_callback(self, callback: Callable[[str], None]) -> None:
        """
        Remove a previously registered consumer.

        Parameters
        ----------
        callback : Callable[[str], None]
            The same callable passed to `register_message_callback()`.
        """
        with self._callback_lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="face-presence-poll", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """
        Request the background thread to stop.

        Parameters
        ----------
        wait : bool
            If True, waits for the thread to finish. Defaults to False.
        """
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
        next_time = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_time:
                time.sleep(min(0.02, next_time - now))
                continue
            try:
                snap = self._fetch_snapshot()
                text = snap.to_text()
                self._emit(text)
            except Exception as e:
                logging.warning(f"Failed to fetch/emit face presence snapshot: {e}")

            next_time += self.period
            if next_time < time.time() - self.period:
                next_time = time.time()

    def _emit(self, text: str) -> None:
        """
        Deliver one formatted presence line to all subscribers.

        Parameters
        ----------
        text : str
            A concise, human-readable snapshot (e.g., "present=[alice], unknown=0, ts=...").
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(text)
            except Exception as e:
                logging.warning(f"Face presence callback failed: {e}")

    async def fetch_snapshot(self, recent_sec: Optional[float] = None) -> PresenceSnapshot:
        """
        Async wrapper for `_fetch_snapshot()`.

        Parameters
        ----------
        recent_sec : Optional[float]
            Lookback window in seconds (overrides self.recent_sec if given).

        Returns
        -------
        PresenceSnapshot
            The canonical presence snapshot.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_snapshot, recent_sec)

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

        faces = data.get("faces", []) or []
        filtered_faces = [f for f in faces if f.get("area", 0) >= self.min_face_area]
        sorted_faces = sorted(filtered_faces, key=lambda f: f.get("area", 0), reverse=True)

        closest_name = "unknown"
        if sorted_faces:
            closest_face_name = (sorted_faces[0].get("name", "") or "").strip()
            if closest_face_name and closest_face_name.lower() != "unknown":
                closest_name = closest_face_name

        seen_names = set()
        names = []
        for face in sorted_faces:
            name = (face.get("name", "") or "").strip()
            if name and name.lower() != "unknown" and name not in seen_names:
                seen_names.add(name)
                names.append(name)

        unknown_count_in_faces = sum(
            1 for face in filtered_faces if (face.get("name", "") or "").strip().lower() == "unknown"
        )

        timestamp = float(data.get("server_ts", time.time()))
        self._unknown_faces = int(unknown_count_in_faces)

        return PresenceSnapshot(
            timestamp=timestamp,
            names=names,
            unknown_faces=unknown_count_in_faces,
            closest_name=closest_name,
            raw=data,
        )

    @property
    def unknown_faces(self) -> int:
        """Most recent (suppressed) count of unknown faces detected in the lookback window."""
        return self._unknown_faces
