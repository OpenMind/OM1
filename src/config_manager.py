from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import json5
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_by_path(data: Dict[str, Any], path: str) -> Any:
    """
    Get value from dict using dot-notation path, e.g. 'a.b.c'.
    """
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class HotReloadSpec:
    enabled: bool
    fields: Tuple[str, ...]


def resolve_hot_reload_spec(config: Dict[str, Any]) -> HotReloadSpec:
    """
    Resolve hot-reload configuration from config + environment variables.
    """
    env_enabled = os.getenv("HOT_RELOAD_ENABLED", "").lower() in {"1", "true", "yes", "on"}

    hot = config.get("hot_reload", {})
    if not isinstance(hot, dict):
        hot = {}

    enabled = bool(hot.get("enabled", env_enabled))
    fields_raw = hot.get("fields", [])

    if not isinstance(fields_raw, list):
        fields_raw = []

    fields = tuple(f for f in fields_raw if isinstance(f, str))

    return HotReloadSpec(enabled=enabled, fields=fields)


class ConfigHotReloader:
    """
    Handles selective hot-reload of whitelisted config fields.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        runtime: Any,
        spec: HotReloadSpec,
        on_applied: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config_path = config_path
        self.runtime = runtime
        self.spec = spec
        self.on_applied = on_applied

        self._last_config: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        text = self.config_path.read_text(encoding="utf-8")
        return json5.loads(text)

    def initialize(self) -> None:
        self._last_config = self._load_config()

    def _compute_patch(
        self, old: Dict[str, Any], new: Dict[str, Any]
    ) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}

        for field in self.spec.fields:
            try:
                old_val = _get_by_path(old, field)
                new_val = _get_by_path(new, field)
            except KeyError:
                continue

            if old_val != new_val:
                patch[field] = new_val

        return patch

    def _apply_patch(self, patch: Dict[str, Any]) -> bool:
        if not patch:
            return False

        # Preferred explicit API
        apply_fn = getattr(self.runtime, "apply_hot_reload", None)
        if callable(apply_fn):
            apply_fn(patch)
            if self.on_applied:
                self.on_applied(patch)
            return True

        # Fallback for common case: system_prompt_base
        if "system_prompt_base" in patch:
            new_prompt = patch["system_prompt_base"]

            llm = getattr(self.runtime, "llm", None)
            if llm is not None:
                if hasattr(llm, "set_system_prompt_base"):
                    llm.set_system_prompt_base(new_prompt)
                    return True
                if hasattr(llm, "system_prompt_base"):
                    setattr(llm, "system_prompt_base", new_prompt)
                    return True

        return False

    def reload_if_changed(self) -> None:
        if not self.spec.enabled:
            return

        if self._last_config is None:
            self.initialize()
            return

        try:
            new_cfg = self._load_config()
        except Exception:
            logger.exception("Hot-reload: invalid JSON5, ignoring change")
            return

        patch = self._compute_patch(self._last_config, new_cfg)
        if not patch:
            return

        if self._apply_patch(patch):
            logger.info("Hot-reload applied: %s", patch)
            self._last_config = new_cfg
        else:
            logger.warning("Hot-reload patch not applied: %s", patch)


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(
        self,
        *,
        target: Path,
        debounce_ms: int,
        callback: Callable[[], None],
    ) -> None:
        self.target = target.resolve()
        self.debounce_ms = debounce_ms
        self.callback = callback

        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _trigger(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(
                self.debounce_ms / 1000.0, self.callback
            )
            self._timer.daemon = True
            self._timer.start()

    def _is_target(self, event_path: str) -> bool:
        try:
            return Path(event_path).resolve() == self.target
        except Exception:
            return False

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_target(event.src_path):
            self._trigger()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_target(event.src_path):
            self._trigger()

    def on_moved(self, event: FileSystemEvent) -> None:
        if hasattr(event, "dest_path") and self._is_target(event.dest_path):
            self._trigger()


class ConfigWatcher:
    def __init__(
        self,
        *,
        config_path: Path,
        reloader: ConfigHotReloader,
        debounce_ms: int = 300,
    ) -> None:
        self.config_path = config_path
        self.reloader = reloader
        self.debounce_ms = debounce_ms

        self._observer: Optional[Observer] = None

    def start(self) -> None:
        if self._observer:
            return

        handler = _DebouncedHandler(
            target=self.config_path,
            debounce_ms=self.debounce_ms,
            callback=self.reloader.reload_if_changed,
        )

        observer = Observer()
        observer.schedule(handler, str(self.config_path.parent), recursive=False)
        observer.daemon = True
        observer.start()

        self._observer = observer
        logger.info("Config hot-reload watcher started for %s", self.config_path)

    def stop(self) -> None:
        if not self._observer:
            return

        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._observer = None
