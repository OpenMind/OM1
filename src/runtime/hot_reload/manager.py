"""
Hot Reload Manager

Main orchestrator for configuration hot-reload. Coordinates file watching,
change detection, validation, and safe application of configuration updates.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .diff import ConfigDiff, compute_config_diff, get_nested_diff
from .strategies import FieldCategory, ReloadStrategy
from .validator import ConfigValidator, ValidationReport
from .watcher import ConfigFileWatcher

logger = logging.getLogger(__name__)


@dataclass
class ReloadEvent:
    """
    Record of a reload event for history tracking.
    
    Attributes:
        timestamp: When the reload occurred
        diff: The configuration changes
        success: Whether the reload succeeded
        strategy: How the reload was handled
        error: Error message if reload failed
    """
    timestamp: datetime
    diff: ConfigDiff
    success: bool
    strategy: str
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"ReloadEvent({status} {self.timestamp.isoformat()} "
            f"strategy={self.strategy} "
            f"changes={self.diff.changed_fields})"
        )


# Type alias for reload callbacks
ReloadCallback = Callable[["ReloadEvent"], None]


class HotReloadManager:
    """
    Manages hot-reload of configuration files.
    
    Features:
    - Automatic file watching with debouncing
    - Selective field-based reloading
    - Validation before applying changes
    - Rollback on validation failure
    - Event callbacks for reload notifications
    - Reload history tracking
    - Thread-safe operation
    
    Usage:
        manager = HotReloadManager(
            config_path="config/agent.json5",
            apply_callback=my_apply_fn
        )
        manager.on_reload(my_notification_fn)
        manager.start()
        
        # Later...
        manager.trigger_reload()  # Manual reload
        manager.stop()
    """
    
    def __init__(
        self,
        config_path: str,
        apply_callback: Optional[Callable[[Dict[str, Any], bool], None]] = None,
        debounce_seconds: float = 1.0,
        max_history: int = 100,
        enable_validation: bool = True,
    ):
        """
        Initialize the hot reload manager.
        
        Args:
            config_path: Path to the configuration file
            apply_callback: Function to call when applying changes.
                           Signature: (changed_values: dict, requires_restart: bool) -> None
            debounce_seconds: Minimum time between reloads
            max_history: Maximum number of reload events to keep in history
            enable_validation: Whether to validate changes before applying
        """
        self.config_path = Path(config_path).resolve()
        self.apply_callback = apply_callback
        self.enable_validation = enable_validation
        self.max_history = max_history
        
        # Current configuration state
        self._current_config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        
        # Reload callbacks
        self._callbacks: List[ReloadCallback] = []
        self._callbacks_lock = threading.Lock()
        
        # Reload history
        self._history: List[ReloadEvent] = []
        self._history_lock = threading.Lock()
        
        # Components
        self.validator = ConfigValidator()
        self.watcher = ConfigFileWatcher(
            config_path=str(self.config_path),
            on_change=self._on_file_change,
            debounce_seconds=debounce_seconds,
        )
        
        # Load initial config
        self._load_initial_config()
    
    def _load_initial_config(self) -> None:
        """Load the initial configuration from file."""
        try:
            config = self._read_config_file()
            with self._config_lock:
                self._current_config = config
            logger.info(
                f"Loaded initial config from {self.config_path} "
                f"({len(config)} fields)"
            )
        except Exception as e:
            logger.error(f"Failed to load initial config: {e}")
            self._current_config = {}
    
    def _read_config_file(self) -> Dict[str, Any]:
        """Read and parse the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        content = self.config_path.read_text(encoding="utf-8")
        
        # Support both JSON and JSON5
        if self.config_path.suffix == ".json5":
            try:
                import json5
                return json5.loads(content)
            except ImportError:
                # Fall back to stripping comments manually
                lines = []
                for line in content.split("\n"):
                    # Remove single-line comments (simple approach)
                    comment_idx = line.find("//")
                    if comment_idx >= 0:
                        line = line[:comment_idx]
                    lines.append(line)
                content = "\n".join(lines)
        
        return json.loads(content)
    
    def _on_file_change(self, path: str) -> None:
        """Handle file change notification from watcher."""
        start_time = time.time()
        event: Optional[ReloadEvent] = None
        
        try:
            # Read new config
            new_config = self._read_config_file()
            
            with self._config_lock:
                old_config = self._current_config.copy()
            
            # Compute diff
            diff = compute_config_diff(old_config, new_config)
            
            if not diff.has_changes:
                logger.debug("Config file changed but no actual differences detected")
                return
            
            logger.info(f"Config changes detected: {diff}")
            
            # Categorize changes
            categories = FieldCategory.categorize_changes(diff.changed_fields)
            
            # Check if restart is required
            requires_restart = bool(categories[ReloadStrategy.RESTART_REQUIRED])
            
            if requires_restart:
                logger.warning(
                    f"Config changes require restart: "
                    f"{categories[ReloadStrategy.RESTART_REQUIRED]}"
                )
                strategy = "restart_required"
            else:
                strategy = "hot_reload"
            
            # Validate changes if enabled
            if self.enable_validation:
                validation_changes = {
                    field: diff.new_values.get(field, new_config.get(field))
                    for field in diff.changed_fields
                    if field not in FieldCategory.IGNORE_FIELDS
                }
                
                report = self.validator.validate_changes(validation_changes)
                
                if not report.is_valid:
                    logger.error(
                        f"Validation failed, not applying changes: "
                        f"{report.error_messages}"
                    )
                    event = ReloadEvent(
                        timestamp=datetime.now(),
                        diff=diff,
                        success=False,
                        strategy="validation_failed",
                        duration_ms=(time.time() - start_time) * 1000,
                        error="; ".join(report.error_messages)
                    )
                    self._add_to_history(event)
                    self._notify_callbacks(event)
                    return
            
            # Apply changes
            if self.apply_callback:
                # Extract changed values
                changed_values = {
                    field: new_config.get(field)
                    for field in diff.changed_fields
                    if field not in FieldCategory.IGNORE_FIELDS
                }
                self.apply_callback(changed_values, requires_restart)
            
            # Update internal state
            with self._config_lock:
                self._current_config = new_config
            
            duration_ms = (time.time() - start_time) * 1000
            
            event = ReloadEvent(
                timestamp=datetime.now(),
                diff=diff,
                success=True,
                strategy=strategy,
                duration_ms=duration_ms,
            )
            
            logger.info(
                f"Config reload successful ({strategy}) in {duration_ms:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Error during config reload: {e}", exc_info=True)
            event = ReloadEvent(
                timestamp=datetime.now(),
                diff=ConfigDiff(),
                success=False,
                strategy="error",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
        
        if event:
            self._add_to_history(event)
            self._notify_callbacks(event)
    
    def _add_to_history(self, event: ReloadEvent) -> None:
        """Add an event to the reload history."""
        with self._history_lock:
            self._history.append(event)
            # Trim history if needed
            while len(self._history) > self.max_history:
                self._history.pop(0)
    
    def _notify_callbacks(self, event: ReloadEvent) -> None:
        """Notify all registered callbacks of a reload event."""
        with self._callbacks_lock:
            callbacks = self._callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")
    
    # Public API
    
    def start(self) -> None:
        """Start watching for configuration changes."""
        self.watcher.start()
        logger.info(f"Hot reload manager started for {self.config_path}")
    
    def stop(self) -> None:
        """Stop watching for configuration changes."""
        self.watcher.stop()
        logger.info("Hot reload manager stopped")
    
    def trigger_reload(self) -> None:
        """
        Manually trigger a configuration reload.
        
        Useful for CLI integration or programmatic reload.
        """
        self.watcher.trigger_reload()
    
    def on_reload(self, callback: ReloadCallback) -> None:
        """
        Register a callback to be notified of reload events.
        
        Args:
            callback: Function that takes a ReloadEvent
        """
        with self._callbacks_lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: ReloadCallback) -> bool:
        """
        Remove a previously registered callback.
        
        Returns True if the callback was found and removed.
        """
        with self._callbacks_lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False
    
    @property
    def current_config(self) -> Dict[str, Any]:
        """Get a copy of the current configuration."""
        with self._config_lock:
            return self._current_config.copy()
    
    @property
    def history(self) -> List[ReloadEvent]:
        """Get a copy of the reload history."""
        with self._history_lock:
            return self._history.copy()
    
    @property
    def is_running(self) -> bool:
        """Check if the manager is currently running."""
        return self.watcher.is_running
    
    def get_field_value(self, field_name: str) -> Any:
        """Get the current value of a configuration field."""
        with self._config_lock:
            return self._current_config.get(field_name)
    
    def get_reload_stats(self) -> Dict[str, Any]:
        """Get statistics about reload operations."""
        with self._history_lock:
            history = self._history.copy()
        
        if not history:
            return {
                "total_reloads": 0,
                "successful": 0,
                "failed": 0,
                "avg_duration_ms": 0.0,
            }
        
        successful = sum(1 for e in history if e.success)
        durations = [e.duration_ms for e in history if e.success]
        
        return {
            "total_reloads": len(history),
            "successful": successful,
            "failed": len(history) - successful,
            "success_rate": successful / len(history) * 100,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "last_reload": history[-1].timestamp.isoformat() if history else None,
        }
