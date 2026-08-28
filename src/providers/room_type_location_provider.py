import logging
import threading
import time
from collections import Counter, deque
from typing import Optional, Set

import requests

from .io_provider import IOProvider
from .singleton import singleton


@singleton
class RoomTypeLocationProvider:
    """
    Provider that reads the `room_type` dynamic variable from IOProvider in a
    background thread and POSTs it to the map locations API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000/maps/locations/add/slam",
        timeout: int = 5,
        refresh_interval: int = 5,
        map_name: str = "map",
    ):
        """
        Initialize the provider.

        Parameters
        ----------
        base_url : str
            The HTTP endpoint to POST room-based locations to.
            Default is "http://localhost:5000/maps/locations/add/slam".
        timeout : int
            Timeout for HTTP POST requests in seconds.
        refresh_interval : int
            How often to check the IOProvider for a room_type in seconds.
        map_name : str
            The map name to include in the payload.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.refresh_interval = refresh_interval
        self.map_name = map_name

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._posted_room_types: Set[str] = set()

        self._room_history = deque()  # stores (timestamp, room_type)

        self._room_window_seconds = refresh_interval * 6

        # Fraction of window the winning label must occupy to be trusted
        self._room_majority_threshold = 0.7

        # How long the winning label must have been present in that window
        self._room_min_stable_seconds = refresh_interval * 3

        self.io_provider = IOProvider()

        # Allowed room type set (must match the values used elsewhere)
        self._allowed_room_types = {
            "living_room",
            "bedroom",
            "study",
            "kitchen",
            "outdoor",
        }

    def start(self) -> None:
        """
        Start the background thread that monitors room_type and posts it.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("RoomTypeLocationProvider already running")
            return

        if not self.base_url:
            logging.error(
                "RoomTypeLocationProvider missing 'base_url'; "
                "provider will not start."
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info(
            "RoomTypeLocationProvider background thread started "
            f"(base_url: {self.base_url}, refresh: {self.refresh_interval}s)"
        )

    def stop(self) -> None:
        """
        Stop the background thread.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            logging.info("RoomTypeLocationProvider background thread stopped")

    def _run(self) -> None:
        """
        Background loop: periodically check IOProvider for room_type and POST.
        """
        while not self._stop_event.is_set():
            try:
                self._process_room_type()
            except Exception:
                logging.exception("Error in RoomTypeLocationProvider loop")

            # Wait until next cycle or until stop event is set
            self._stop_event.wait(timeout=self.refresh_interval)

    def _get_current_room_type(self) -> Optional[str]:
        """
        Read the current room_type from IOProvider.

        Returns
        -------
        Optional[str]
            The current normalized room type, or None if not set/invalid.
        """
        try:
            room_type = self.io_provider.get_dynamic_variable("room_type")
        except Exception:
            logging.exception("Failed to read 'room_type' from IOProvider")
            return None

        if not room_type:
            return None

        # Ensure it's a string and normalized
        if not isinstance(room_type, str):
            room_type = str(room_type)

        room_type = room_type.strip().lower()

        if room_type not in self._allowed_room_types:
            logging.debug(
                f"Room type '{room_type}' found in dynamic variables, "
                "but not in allowed set; ignoring."
            )
            return None

        return room_type

    def _process_room_type(self) -> None:
        """
        Check the current room type and POST it.
        """
        if not self.base_url:
            # Already logged in start(), just silently return here
            return

        room_type = self._get_current_room_type()
        if not room_type:
            # No valid room type available
            return

        now = time.time()

        # Reading to history
        self._room_history.append((now, room_type))

        # Drop old entries outside the time window
        cutoff = now - self._room_window_seconds
        while self._room_history and self._room_history[0][0] < cutoff:
            self._room_history.popleft()

        if not self._room_history:
            return

        # Majority vote over the window
        counts = Counter(rt for _, rt in self._room_history)
        winner, winner_count = counts.most_common(1)[0]
        total = len(self._room_history)
        majority_fraction = winner_count / float(total)

        if majority_fraction < self._room_majority_threshold:
            logging.debug(
                "RoomTypeLocationProvider: no strong majority yet "
                f"(winner={winner}, frac={majority_fraction:.2f}, total={total})"
            )
            return

        # Check how long this winner has been present
        first_winner_ts = min(ts for ts, rt in self._room_history if rt == winner)
        stable_seconds = now - first_winner_ts

        if stable_seconds < self._room_min_stable_seconds:
            logging.debug(
                "RoomTypeLocationProvider: majority label not yet stable "
                f"(winner={winner}, stable_for={stable_seconds:.1f}s)"
            )
            return

        smoothed_room_type = winner

        # If we've already posted this label, keep the old behaviour: skip
        with self._lock:
            if smoothed_room_type in self._posted_room_types:
                logging.debug(
                    f"RoomTypeLocationProvider: '{smoothed_room_type}' already posted; skipping."
                )
                return

        payload = {
            "map_name": self.map_name,
            "label": smoothed_room_type,
            "description": (
                f"Auto-generated location for room type '{smoothed_room_type}' "
                f"(majority={majority_fraction:.2f}, stable_for={stable_seconds:.1f}s)"
            ),
        }

        try:
            logging.info(
                f"Posting smoothed room_type '{smoothed_room_type}' to "
                f"{self.base_url} with payload: {payload}"
            )
            resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
            text = resp.text

            if 200 <= resp.status_code < 300:
                logging.info(
                    f"Room type location stored '{smoothed_room_type}' -> "
                    f"{resp.status_code} {text}"
                )
                with self._lock:
                    self._posted_room_types.add(smoothed_room_type)
            else:
                logging.error(
                    f"Room type location API returned {resp.status_code}: {text}"
                )

        except requests.Timeout:
            logging.error("Room type location API request timed out")
        except Exception as e:
            logging.error(f"Room type location API request failed: {e}")
