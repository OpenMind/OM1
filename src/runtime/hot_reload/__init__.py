"""
Hot-reload module for selective configuration updates.

This module provides functionality to detect configuration file changes
and apply updates selectively based on the type of field that changed,
avoiding full system restarts when possible.
"""

from runtime.hot_reload.strategies import ReloadStrategy, get_field_strategy
from runtime.hot_reload.diff import ConfigDiff, diff_configs
from runtime.hot_reload.watcher import ConfigFileWatcher
from runtime.hot_reload.manager import HotReloadManager

__all__ = [
    "ReloadStrategy",
    "get_field_strategy",
    "ConfigDiff",
    "diff_configs",
    "ConfigFileWatcher",
    "HotReloadManager",
]
