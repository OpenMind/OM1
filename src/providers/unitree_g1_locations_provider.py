import json
import logging
import threading
from typing import Dict, List, Optional, Union

import requests

from .io_provider import IOProvider
from .singleton import singleton


@singleton
class UnitreeG1LocationsProvider:
    """
    Provider that fetches locations from HTTP API in a background thread for Unitree G1.

    This class implements a singleton pattern to manage:
        * Periodic background fetching of location data from an HTTP API
        * Thread-safe access to the location dictionary

    Parameters
    ----------
    base_url : str, optional
        The HTTP endpoint to fetch locations from. Default is "http://localhost:5000/maps/locations/list".
    timeout : int, optional
        Timeout for HTTP requests in seconds. Default is 5.
    refresh_interval : int, optional
        How often to refresh locations in seconds. Default is 30.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000/maps/locations/list",
        timeout: int = 5,
        refresh_interval: int = 30,
    ):
        """
        Initialize the Unitree G1 Locations Provider with a specific API endpoint and refresh interval.

        Parameters
        ----------
        base_url : str, optional
            The HTTP endpoint to fetch locations from. Default is "http://localhost:5000/maps/locations/list".
        timeout : int, optional
            Timeout for HTTP requests in seconds. Default is 5.
        refresh_interval : int, optional
            How often to refresh locations in seconds. Default is 30.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.refresh_interval = refresh_interval
        self._locations: Dict[str, Dict] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.io_provider = IOProvider()

    def start(self) -> None:
        """
        Start the background fetch thread.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("UnitreeG1LocationsProvider already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("UnitreeG1LocationsProvider background thread started")

    def stop(self) -> None:
        """
        Stop the background fetch thread.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """
        Background thread that periodically fetches locations.
        """
        while not self._stop_event.is_set():
            try:
                self._fetch()
            except Exception:
                logging.exception("Error fetching locations")
            self._stop_event.wait(timeout=self.refresh_interval)

    def _fetch(self) -> None:
        """
        Fetch locations from the API and update cache.
        """
        if not self.base_url:
            return
        try:
            resp = requests.get(self.base_url, timeout=self.timeout)
            if resp.status_code < 200 or resp.status_code >= 300:
                logging.error(
                    f"Location list API returned {resp.status_code}: {resp.text}"
                )
                return
            data = resp.json()
            raw_message = data.get("message") if isinstance(data, dict) else None
            if raw_message and isinstance(raw_message, str):
                try:
                    locations = json.loads(raw_message)
                except Exception:
                    logging.error(
                        "Failed to parse nested message JSON from location list"
                    )
                    return
            elif isinstance(data, dict) and "message" not in data:
                locations = data
            else:
                logging.error("Unexpected format from location list API")
                return
            self._update_locations(locations)
        except Exception:
            logging.exception("Error fetching locations")

    def _update_locations(self, locations_raw: Union[Dict, List]) -> None:
        """
        Parse and store locations.

        Parameters
        ----------
        locations_raw : Dict or List
            Raw locations data from the API.
        """
        parsed = {}
        if isinstance(locations_raw, dict):
            for k, v in locations_raw.items():
                entry = v if isinstance(v, dict) else {"name": k, "pose": {}}
                entry.setdefault("name", k)
                parsed[k.strip().lower()] = entry
        elif isinstance(locations_raw, list):
            for item in locations_raw:
                if not isinstance(item, dict):
                    continue
                name = (item.get("name") or item.get("label") or "").strip()
                if not name:
                    continue
                parsed[name.lower()] = item
        with self._lock:
            self._locations = parsed

    def get_all_locations(self) -> Dict[str, Dict]:
        """
        Get all cached locations.

        Returns
        -------
        Dict
            A dictionary of all locations keyed by their labels.
        """
        with self._lock:
            return dict(self._locations)

    def get_location(self, label: str) -> Optional[Dict]:
        """
        Get a specific location by label.

        Parameters
        ----------
        label : str
            The label of the location to retrieve.

        Returns
        -------
        Dict or None
            The location data if found, otherwise None.
        """
        if not label:
            return None
        key = label.strip().lower()
        with self._lock:
            return self._locations.get(key)
