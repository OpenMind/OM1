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
        Set the asyncio event loop used to schedule callbacks
        from the filesystem watcher thread.
        """
        self._loop = loop

    def on_modified(self, event):
        """
        Handle filesystem modification events.

        Triggers the registered callback when the watched configuration
        file is modified.
        """
        if event.is_directory:
            return

        event_path = Path(event.src_path).resolve()
        if event_path != self.target_path:
            return

        # Check if file still exists
        if not self.target_path.exists():
            logging.warning(f"Config file deleted: {self.target_path}")
            return

        now = time.time()
        if now - self._last_modified < self.debounce_seconds:
            logging.debug(
                f"Debouncing file change: {event_path} "
                f"(last mod: {now - self._last_modified:.2f}s ago)"
            )
            return

        self._last_modified = now
        logging.info(f"Config file modified: {event_path}")

        if self._loop and self._loop.is_running():
            # --- PERUBAHAN: Bungkus callback dalam fungsi async untuk mendapatkan Coroutine ---
            async def _wrapper():
                await self.callback(event_path)

            asyncio.run_coroutine_threadsafe(
                _wrapper(),  # Panggil _wrapper() untuk mendapatkan objek Coroutine
                self._loop,
            )
            # --- PERUBAHAN SELESAI ---
        else:
            logging.warning("Event loop not available, cannot trigger callback")


class ConfigFileWatcher:
    """
    Watch configuration files for changes using watchdog.

    Supports both legacy and new APIs:
    - legacy: callback(path)
    - new: on_change_callback(...)
    """

    def __init__(
        self,
        config_path: Path,
        callback: Callable[[Path], Awaitable[None]] | None = None,
        debounce_seconds: float = 0.5,
        *,
        on_change_callback: Callable[..., Awaitable[None]] | None = None,
    ):
        if on_change_callback is None:
            on_change_callback = callback

        self.config_path = Path(config_path).resolve()
        self._on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds

        self._observer: Optional[Observer] = None
        self._handler: Optional[ConfigFileHandler] = None
        self._watching = False

        if not self.config_path.exists():
            logging.warning(f"Config file does not exist yet: {self.config_path}")

    async def _handle_file_change(self, path: Path) -> None:
        if self._on_change_callback is None:
            return
        try:
            await self._on_change_callback(path)
        except Exception as e:
            logging.error(f"Hot-reload callback failed: {e}")

    def start(self, event_loop: asyncio.AbstractEventLoop | None = None):
        """
        Start watching the configuration file.

        event_loop is optional for backward compatibility.
        """
        if self._watching:
            logging.warning("File watcher already started")
            return

        if event_loop is None:
            event_loop = asyncio.get_event_loop()

        self._handler = ConfigFileHandler(
            callback=self._handle_file_change,
            target_path=self.config_path,
            debounce_seconds=self.debounce_seconds,
        )
        self._handler.set_event_loop(event_loop)

        self._observer = Observer()
        watch_dir = self.config_path.parent
        # --- PERUBAHAN: Tambahkan assert untuk memberi tahu pyright bahwa _observer bukan None ---
        assert self._observer is not None, "Observer should be initialized here"
        self._observer.schedule(self._handler, str(watch_dir), recursive=False)
        self._observer.start()
        # --- PERUBAHAN SELESAI ---

        self._watching = True
        logging.info(
            f"Started watching config file: {self.config_path} "
            f"(debounce: {self.debounce_seconds}s)"
        )

    def stop(self):
        """
        Stop watching the configuration file and release resources.
        """
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
        Return whether the configuration watcher is currently active.
        """
        return self._watching
