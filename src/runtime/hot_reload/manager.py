"""
Hot-reload manager for selective configuration updates.

This module provides the main HotReloadManager class that orchestrates
configuration file watching, change detection, and selective reloading.
"""

import asyncio
import copy
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import json5

from runtime.hot_reload.diff import ConfigDiff, diff_configs
from runtime.hot_reload.strategies import (
    ReloadStrategy,
    categorize_changes,
    get_field_strategy,
    validate_field,
)
from runtime.hot_reload.watcher import ConfigFileWatcher


@dataclass
class ReloadResult:
    """
    Result of a reload operation.

    Attributes
    ----------
    success : bool
        Whether the reload was successful.
    hot_reloaded_fields : Set[str]
        Fields that were hot-reloaded.
    requires_restart : bool
        Whether a full restart is required.
    restart_fields : Set[str]
        Fields that require restart.
    errors : List[str]
        Any errors that occurred.
    rollback_performed : bool
        Whether a rollback was performed due to errors.
    """

    success: bool = True
    hot_reloaded_fields: Set[str] = field(default_factory=set)
    requires_restart: bool = False
    restart_fields: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    rollback_performed: bool = False


@dataclass
class ChangeHistoryEntry:
    """
    Record of a configuration change.

    Attributes
    ----------
    timestamp : float
        When the change occurred.
    diff : ConfigDiff
        The detected changes.
    result : ReloadResult
        The result of the reload operation.
    """

    timestamp: float
    diff: ConfigDiff
    result: ReloadResult


class HotReloadManager:
    """
    Manages selective hot-reloading of configuration fields.

    This class coordinates between the file watcher, diff engine, and
    reload strategies to provide intelligent configuration updates that
    minimize system disruption.

    Thread Safety
    -------------
    This class is thread-safe. All shared state is protected by locks,
    and callbacks from the file watcher are properly synchronized with
    the asyncio event loop.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    on_hot_reload : Optional[Callable]
        Callback for hot-reloadable field changes.
        Signature: (field_name: str, old_value: Any, new_value: Any) -> bool
    on_restart_required : Optional[Callable]
        Callback when restart-required fields change.
        Signature: (changed_fields: Dict[str, tuple]) -> None
    debounce_seconds : float
        Debounce time for file changes (default: 0.5s).
    max_history : int
        Maximum number of change history entries to keep (default: 50).
    """

    def __init__(
        self,
        config_path: str,
        on_hot_reload: Optional[Callable[[str, Any, Any], bool]] = None,
        on_restart_required: Optional[Callable[[Dict[str, tuple]], None]] = None,
        debounce_seconds: float = 0.5,
        max_history: int = 50,
    ):
        self.config_path = os.path.abspath(config_path)
        self.on_hot_reload = on_hot_reload
        self.on_restart_required = on_restart_required
        self.debounce_seconds = debounce_seconds
        self.max_history = max_history

        # Thread safety
        self._lock = threading.RLock()

        # State
        self._current_config: Optional[Dict[str, Any]] = None
        self._original_config: Optional[Dict[str, Any]] = None
        self._watcher: Optional[ConfigFileWatcher] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = False

        # Change tracking
        self._change_history: List[ChangeHistoryEntry] = []
        self._pending_restart_fields: Set[str] = set()

        logging.debug(f"HotReloadManager initialized for: {self.config_path}")

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file.

        Returns
        -------
        Optional[Dict[str, Any]]
            The loaded configuration, or None if loading failed.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json5.load(f)
        except json5.JSON5DecodeError as e:
            logging.error(f"Failed to parse config: {e}")
            return None
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return None

    async def start(
        self, event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> bool:
        """
        Start the hot-reload manager.

        Parameters
        ----------
        event_loop : Optional[asyncio.AbstractEventLoop]
            The event loop to use. If None, gets the running loop.

        Returns
        -------
        bool
            True if started successfully.
        """
        with self._lock:
            if self._is_running:
                logging.warning("HotReloadManager is already running")
                return True

            # Get event loop
            self._event_loop = event_loop or asyncio.get_running_loop()

            # Load initial config
            self._current_config = self._load_config()
            if self._current_config is None:
                logging.error("Failed to load initial configuration")
                return False

            # Keep a copy of original config for rollback
            self._original_config = copy.deepcopy(self._current_config)

            # Create and start watcher
            self._watcher = ConfigFileWatcher(self.config_path, self.debounce_seconds)
            self._watcher.set_async_callback(self._on_config_changed, self._event_loop)

            if not self._watcher.start():
                logging.error("Failed to start config file watcher")
                return False

            self._is_running = True
            logging.info(f"HotReloadManager started for: {self.config_path}")
            return True

    def stop(self) -> None:
        """Stop the hot-reload manager."""
        with self._lock:
            if not self._is_running:
                return

            if self._watcher:
                self._watcher.stop()
                self._watcher = None

            self._is_running = False
            logging.info("HotReloadManager stopped")

    async def _on_config_changed(self, file_path: str) -> None:
        """
        Handle configuration file changes.

        This is called by the file watcher when changes are detected.

        Parameters
        ----------
        file_path : str
            Path to the changed file.
        """
        logging.info(f"Configuration change detected: {file_path}")

        try:
            result = await self.check_and_reload()

            if result.success:
                if result.hot_reloaded_fields:
                    logging.info(f"Hot-reloaded fields: {result.hot_reloaded_fields}")
                if result.requires_restart:
                    logging.warning(
                        f"Restart required for fields: {result.restart_fields}"
                    )
            else:
                logging.error(f"Reload failed: {result.errors}")

        except Exception as e:
            logging.error(f"Error handling config change: {e}")

    async def check_and_reload(self) -> ReloadResult:
        """
        Check for configuration changes and apply them.

        This method:
        1. Loads the new configuration
        2. Computes the diff with current config
        3. Categorizes changes by reload strategy
        4. Applies hot-reloadable changes
        5. Tracks restart-required changes

        Returns
        -------
        ReloadResult
            The result of the reload operation.
        """
        result = ReloadResult()

        with self._lock:
            # Load new config
            new_config = self._load_config()
            if new_config is None:
                result.success = False
                result.errors.append("Failed to load new configuration")
                return result

            if self._current_config is None:
                self._current_config = new_config
                return result

            # Compute diff
            diff = diff_configs(self._current_config, new_config)

            if not diff.has_changes:
                logging.debug("No configuration changes detected")
                return result

            logging.info(f"Detected {len(diff.changes)} config changes")

            # Categorize changes
            categorized = categorize_changes(diff.changed_fields)

            # Process hot-reloadable changes
            hot_reload_changes = {
                **categorized[ReloadStrategy.HOT_RELOAD],
                **categorized[ReloadStrategy.VALIDATE_FIRST],
            }

            for field_name, (old_val, new_val) in hot_reload_changes.items():
                # Validate if needed
                strategy = get_field_strategy(field_name)
                if strategy == ReloadStrategy.VALIDATE_FIRST:
                    if not validate_field(field_name, old_val, new_val):
                        result.errors.append(
                            f"Validation failed for field '{field_name}'"
                        )
                        continue

                # Apply the change
                success = await self._apply_hot_reload(field_name, old_val, new_val)
                if success:
                    result.hot_reloaded_fields.add(field_name)
                    # Update current config
                    self._set_config_value(field_name, new_val)
                else:
                    result.errors.append(f"Failed to hot-reload field '{field_name}'")

            # Track restart-required changes
            restart_changes = categorized[ReloadStrategy.RESTART_REQUIRED]
            if restart_changes:
                result.requires_restart = True
                result.restart_fields = set(restart_changes.keys())
                self._pending_restart_fields.update(restart_changes.keys())

                # Notify callback
                if self.on_restart_required:
                    try:
                        self.on_restart_required(restart_changes)
                    except Exception as e:
                        logging.error(f"Error in restart callback: {e}")

            # Record history
            self._add_history_entry(diff, result)

            result.success = len(result.errors) == 0

        return result

    async def _apply_hot_reload(
        self, field_name: str, old_value: Any, new_value: Any
    ) -> bool:
        """
        Apply a hot-reload change for a single field.

        Parameters
        ----------
        field_name : str
            The field to update.
        old_value : Any
            The current value.
        new_value : Any
            The new value.

        Returns
        -------
        bool
            True if the change was applied successfully.
        """
        logging.debug(f"Hot-reloading field '{field_name}': {old_value} -> {new_value}")

        if self.on_hot_reload:
            try:
                return self.on_hot_reload(field_name, old_value, new_value)
            except Exception as e:
                logging.error(f"Error in hot-reload callback for '{field_name}': {e}")
                return False

        # Default: just accept the change
        return True

    def _set_config_value(self, field_path: str, value: Any) -> None:
        """Set a value in the current config."""
        if self._current_config is None:
            return

        parts = field_path.split(".")
        current = self._current_config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _add_history_entry(self, diff: ConfigDiff, result: ReloadResult) -> None:
        """Add an entry to the change history."""
        entry = ChangeHistoryEntry(timestamp=time.time(), diff=diff, result=result)
        self._change_history.append(entry)

        # Trim history if needed
        if len(self._change_history) > self.max_history:
            self._change_history = self._change_history[-self.max_history :]

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current configuration.

        Returns
        -------
        Optional[Dict[str, Any]]
            Copy of the current configuration.
        """
        with self._lock:
            if self._current_config is None:
                return None
            return copy.deepcopy(self._current_config)

    def get_pending_restart_fields(self) -> Set[str]:
        """
        Get fields that require restart but haven't been applied.

        Returns
        -------
        Set[str]
            Set of field names requiring restart.
        """
        with self._lock:
            return self._pending_restart_fields.copy()

    def clear_pending_restart_fields(self) -> None:
        """Clear the pending restart fields (after restart)."""
        with self._lock:
            self._pending_restart_fields.clear()

    def get_change_history(self) -> List[ChangeHistoryEntry]:
        """
        Get the change history.

        Returns
        -------
        List[ChangeHistoryEntry]
            List of historical changes (newest last).
        """
        with self._lock:
            return self._change_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hot-reload manager.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing various statistics.
        """
        with self._lock:
            return {
                "is_running": self._is_running,
                "config_path": self.config_path,
                "history_count": len(self._change_history),
                "pending_restart_fields": list(self._pending_restart_fields),
                "debounce_seconds": self.debounce_seconds,
            }

    async def force_reload(self) -> ReloadResult:
        """
        Force a reload check regardless of file changes.

        Returns
        -------
        ReloadResult
            The result of the reload operation.
        """
        return await self.check_and_reload()

    def rollback_to_original(self) -> bool:
        """
        Rollback to the original configuration.

        Returns
        -------
        bool
            True if rollback was successful.
        """
        with self._lock:
            if self._original_config is None:
                logging.error("No original config to rollback to")
                return False

            self._current_config = copy.deepcopy(self._original_config)
            self._pending_restart_fields.clear()
            logging.info("Configuration rolled back to original")
            return True

    @property
    def is_running(self) -> bool:
        """Check if the manager is running."""
        with self._lock:
            return self._is_running

    def __enter__(self) -> "HotReloadManager":
        """Context manager entry (for sync usage with manual start)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    async def __aenter__(self) -> "HotReloadManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.stop()
