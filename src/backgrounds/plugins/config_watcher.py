"""
Config Watcher Background
监控配置文件变化并自动重载
"""

import logging
import os
import threading
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from backgrounds.base import Background, BackgroundConfig


class ConfigWatcher(Background):
    """
    Monitor configuration file changes and trigger hot-reload.
    
    Automatically detects changes to configuration files and
    triggers reload without requiring system restart.
    
    Usage:
        Add to config backgrounds with path to config file(s) to watch.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.config_paths = getattr(config, "config_paths", [])
        self.reload_callback = getattr(config, "reload_callback", None)
        self.debounce_seconds = getattr(config, "debounce_seconds", 2.0)

        if not self.config_paths:
            logging.warning("ConfigWatcher: No config_paths specified")
            return

        self._running = True
        self.last_modified = {}
        self.observer = Observer()
        self.handler = ConfigFileHandler(self)

        # Watch each config file
        for path in self.config_paths:
            full_path = Path(path).expanduser().resolve()
            if full_path.exists():
                watch_dir = str(full_path.parent)
                self.observer.schedule(self.handler, watch_dir, recursive=False)
                logging.info(f"ConfigWatcher: Watching {full_path}")
            else:
                logging.warning(f"ConfigWatcher: Path not found: {full_path}")

        self.observer.start()
        logging.info("✅ ConfigWatcher: Started")

    def on_config_changed(self, file_path):
        """Handle configuration file change."""
        current_time = time.time()
        last_time = self.last_modified.get(file_path, 0)

        # Debounce rapid changes
        if current_time - last_time < self.debounce_seconds:
            return

        self.last_modified[file_path] = current_time
        logging.info(f"📝 Config changed: {file_path}")

        # Trigger reload callback
        if self.reload_callback:
            try:
                self.reload_callback(file_path)
            except Exception as e:
                logging.error(f"Config reload failed: {e}")

    def stop(self):
        """Stop watching files."""
        self._running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()


class ConfigFileHandler(FileSystemEventHandler):
    """Handle file system events for configuration files."""

    def __init__(self, watcher):
        self.watcher = watcher

    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return

        # Check if modified file is in our watch list
        for path in self.watcher.config_paths:
            full_path = Path(path).expanduser().resolve()
            if Path(event.src_path).resolve() == full_path:
                self.watcher.on_config_changed(str(full_path))
                break
