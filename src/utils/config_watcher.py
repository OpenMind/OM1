"""
Configuration file watcher using watchdog library.

Provides real-time file monitoring for configuration changes
with debouncing and async callback support.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ConfigFileHandler(FileSystemEventHandler):
    """
    Handle file system events for configuration files.

    Parameters
    ----------
    callback : Callable
        Async callback function to invoke on file changes
    target_path : Path
        Path to the configuration file to monitor
    debounce_seconds : float
        Minimum time between triggers for the same file
    """

    def __init__(
        self,
        callback: Callable[[Path], Awaitable[None]],
        target_path: Path,
        debounce_seconds: float = 0.5,
    ):
        super().__init__()
        self.callback = callback
        self.target_path = target_path.resolve()
        self.debounce_seconds = debounce_seconds
        self._last_modified = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Set the event loop for async callbacks.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            Event loop to use for scheduling callbacks
        """
        self._loop = loop

    def on_modified(self, event):
        """
        Handle file modification events.

        Parameters
        ----------
        event : FileSystemEvent
            File system event that triggered this handler
        """
        if event.is_directory:
            return

        # Only react to our target file
        event_path = Path(event.src_path).resolve()
        if event_path != self.target_path:
            return

        # Debounce: ignore rapid successive modifications
        now = time.time()
        if now - self._last_modified < self.debounce_seconds:
            logging.debug(
                f"Debouncing file change: {event_path} "
                f"(last mod: {now - self._last_modified:.2f}s ago)"
            )
            return

        self._last_modified = now
        logging.info(f"Config file modified: {event_path}")

        # Schedule callback in event loop
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.callback(event_path), self._loop)
        else:
            logging.warning("Event loop not available, cannot trigger callback")


class ConfigFileWatcher:
    """
    Watch configuration files for changes using watchdog.

    This watcher monitors configuration files and triggers async callbacks
    when changes are detected, with built-in debouncing to avoid multiple
    triggers from single save operations.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file to watch
    callback : Callable
        Async callback function(path) to invoke on changes
    debounce_seconds : float, optional
        Minimum time between triggers (default: 0.5)
    """

    def __init__(
        self,
        config_path: Path,
        callback: Callable[[Path], Awaitable[None]],
        debounce_seconds: float = 0.5,
    ):
        self.config_path = Path(config_path).resolve()
        self.callback = callback
        self.debounce_seconds = debounce_seconds

        self._observer: Optional[Observer] = None
        self._handler: Optional[ConfigFileHandler] = None
        self._watching = False

        # Verify file exists
        if not self.config_path.exists():
            logging.warning(f"Config file does not exist yet: {self.config_path}")

    def start(self, event_loop: asyncio.AbstractEventLoop):
        """
        Start watching the configuration file.

        Parameters
        ----------
        event_loop : asyncio.AbstractEventLoop
            Event loop to use for async callbacks
        """
        if self._watching:
            logging.warning("File watcher already started")
            return

        # Create handler
        self._handler = ConfigFileHandler(
            callback=self.callback,
            target_path=self.config_path,
            debounce_seconds=self.debounce_seconds,
        )
        self._handler.set_event_loop(event_loop)

        # Create observer
        self._observer = Observer()

        # Watch the directory containing the config file
        watch_dir = self.config_path.parent
        self._observer.schedule(self._handler, str(watch_dir), recursive=False)

        # Start observer
        self._observer.start()
        self._watching = True

        logging.info(
            f"Started watching config file: {self.config_path} "
            f"(debounce: {self.debounce_seconds}s)"
        )

    def stop(self):
        """Stop watching the configuration file."""
        if not self._watching:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

        self._watching = False
        self._observer = None
        self._handler = None

        logging.info(f"Stopped watching config file: {self.config_path}")

    def is_watching(self) -> bool:
        """
        Check if watcher is currently active.

        Returns
        -------
        bool
            True if watcher is active
        """
        return self._watching
