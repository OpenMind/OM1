"""
File watcher for configuration hot-reload using watchdog.

This module provides event-driven file monitoring instead of polling,
which is more efficient and responsive to configuration changes.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import json5
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class ConfigFileHandler(FileSystemEventHandler):
    """
    Handles file system events for configuration files.

    This handler implements debouncing to avoid multiple rapid callbacks
    when editors save files (which often involves multiple write operations).
    """

    def __init__(
        self,
        config_path: str,
        callback: Callable[[str], None],
        debounce_seconds: float = 0.5,
    ):
        """
        Initialize the file handler.

        Parameters
        ----------
        config_path : str
            Path to the configuration file to watch.
        callback : Callable[[str], None]
            Function to call when file changes (receives file path).
        debounce_seconds : float
            Minimum time between callbacks to avoid duplicate triggers.
        """
        super().__init__()
        self.config_path = os.path.abspath(config_path)
        self.config_filename = os.path.basename(config_path)
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._last_modified: float = 0.0
        self._lock = threading.Lock()

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if this is our config file
        event_path = os.path.abspath(event.src_path)

        # Match by filename since some editors create temp files
        if os.path.basename(event_path) != self.config_filename:
            return

        # Debounce: ignore rapid successive modifications
        with self._lock:
            current_time = time.time()
            if current_time - self._last_modified < self.debounce_seconds:
                logging.debug(f"Debouncing config change for {self.config_filename}")
                return
            self._last_modified = current_time

        logging.info(f"Config file modified: {self.config_filename}")

        try:
            self.callback(self.config_path)
        except Exception as e:
            logging.error(f"Error in config change callback: {e}")


class ConfigFileWatcher:
    """
    Watches a configuration file for changes using watchdog.

    This class provides an event-driven approach to detecting configuration
    changes, which is more efficient than polling-based approaches.

    Thread Safety
    -------------
    This class is thread-safe. The watchdog observer runs in a separate
    thread, and all shared state is protected by locks.
    """

    def __init__(self, config_path: str, debounce_seconds: float = 0.5):
        """
        Initialize the configuration file watcher.

        Parameters
        ----------
        config_path : str
            Path to the configuration file to watch.
        debounce_seconds : float
            Minimum time between change notifications (default: 0.5s).
        """
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        self.debounce_seconds = debounce_seconds

        self._observer: Optional[Observer] = None
        self._handler: Optional[ConfigFileHandler] = None
        self._callback: Optional[Callable[[str], None]] = None
        self._async_callback: Optional[Callable[[str], Any]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        self._lock = threading.RLock()
        self._is_running = False

        # Store last known config for comparison
        self._last_config: Optional[Dict[str, Any]] = None
        self._last_mtime: float = 0.0

        logging.debug(f"ConfigFileWatcher initialized for: {self.config_path}")

    def _sync_callback_wrapper(self, file_path: str) -> None:
        """
        Wrapper to handle both sync and async callbacks.

        This is called from the watchdog thread and needs to properly
        dispatch to the async callback if one is set.
        """
        if self._async_callback and self._event_loop:
            try:
                # Schedule the async callback on the event loop
                asyncio.run_coroutine_threadsafe(
                    self._async_callback(file_path), self._event_loop
                )
            except Exception as e:
                logging.error(f"Error scheduling async callback: {e}")
        elif self._callback:
            try:
                self._callback(file_path)
            except Exception as e:
                logging.error(f"Error in sync callback: {e}")

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set a synchronous callback for file changes.

        Parameters
        ----------
        callback : Callable[[str], None]
            Function to call when file changes.
        """
        with self._lock:
            self._callback = callback

    def set_async_callback(
        self, callback: Callable[[str], Any], event_loop: asyncio.AbstractEventLoop
    ) -> None:
        """
        Set an asynchronous callback for file changes.

        Parameters
        ----------
        callback : Callable[[str], Any]
            Async function to call when file changes.
        event_loop : asyncio.AbstractEventLoop
            The event loop to schedule the callback on.
        """
        with self._lock:
            self._async_callback = callback
            self._event_loop = event_loop

    def start(self) -> bool:
        """
        Start watching the configuration file.

        Returns
        -------
        bool
            True if started successfully, False otherwise.
        """
        with self._lock:
            if self._is_running:
                logging.warning("ConfigFileWatcher is already running")
                return True

            if not os.path.exists(self.config_path):
                logging.error(f"Config file does not exist: {self.config_path}")
                return False

            try:
                # Load initial config state
                self._load_current_config()

                # Create handler and observer
                self._handler = ConfigFileHandler(
                    self.config_path, self._sync_callback_wrapper, self.debounce_seconds
                )

                self._observer = Observer()
                self._observer.schedule(self._handler, self.config_dir, recursive=False)
                self._observer.start()

                self._is_running = True
                logging.info(f"Started watching config file: {self.config_path}")
                return True

            except Exception as e:
                logging.error(f"Failed to start config watcher: {e}")
                self._cleanup()
                return False

    def stop(self) -> None:
        """Stop watching the configuration file."""
        with self._lock:
            if not self._is_running:
                return

            self._cleanup()
            self._is_running = False
            logging.info("Config file watcher stopped")

    def _cleanup(self) -> None:
        """Clean up observer and handler."""
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception as e:
                logging.warning(f"Error stopping observer: {e}")
            self._observer = None

        self._handler = None

    def _load_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Load and cache the current configuration.

        Returns
        -------
        Optional[Dict[str, Any]]
            The loaded configuration, or None if loading failed.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json5.load(f)

            self._last_config = config
            self._last_mtime = os.path.getmtime(self.config_path)
            return config

        except json5.JSON5DecodeError as e:
            logging.error(f"Failed to parse config file: {e}")
            return None
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            return None

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current configuration from the file.

        This always reads fresh from disk.

        Returns
        -------
        Optional[Dict[str, Any]]
            The current configuration, or None if loading failed.
        """
        return self._load_current_config()

    def get_last_known_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the last successfully loaded configuration.

        Returns
        -------
        Optional[Dict[str, Any]]
            The cached configuration.
        """
        with self._lock:
            return self._last_config

    def has_file_changed(self) -> bool:
        """
        Check if the file has been modified since last load.

        Returns
        -------
        bool
            True if file modification time has changed.
        """
        try:
            current_mtime = os.path.getmtime(self.config_path)
            return current_mtime > self._last_mtime
        except OSError:
            return False

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        with self._lock:
            return self._is_running

    def __enter__(self) -> "ConfigFileWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.stop()
