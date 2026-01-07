import threading
from typing import Any, Callable, Dict

import json5
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def _diff_dict(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict containing only keys whose values changed.
    Nested dicts are diffed recursively.
    """
    patch: Dict[str, Any] = {}

    for key, new_value in new.items():
        if key not in old:
            patch[key] = new_value
            continue

        old_value = old[key]

        if isinstance(old_value, dict) and isinstance(new_value, dict):
            sub = _diff_dict(old_value, new_value)
            if sub:
                patch[key] = sub
        else:
            if old_value != new_value:
                patch[key] = new_value

    return patch


class _ConfigFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        path: str,
        load_fn: Callable[[], Dict[str, Any]],
        on_patch: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._path = path
        self._load_fn = load_fn
        self._on_patch = on_patch
        self._last_config = load_fn()

    def on_modified(self, event) -> None:
        if event.src_path != self._path:
            return

        try:
            new_config = self._load_fn()
        except Exception:
            # ignore transient / partial writes
            return

        patch = _diff_dict(self._last_config, new_config)
        if patch:
            self._last_config = new_config
            self._on_patch(patch)


class ConfigManager:
    """
    Watches a JSON5 config file and emits patches on change.
    """

    def __init__(
        self,
        path: str,
        on_patch: Callable[[Dict[str, Any]], None],
        poll_interval: float = 0.25,
    ) -> None:
        self._path = path
        self._on_patch = on_patch
        self._poll_interval = poll_interval
        self._observer: Observer | None = None

    def _load(self) -> Dict[str, Any]:
        with open(self._path, "r", encoding="utf-8") as f:
            return json5.load(f)

    def start(self) -> None:
        if self._observer is not None:
            return

        handler = _ConfigFileHandler(
            path=self._path,
            load_fn=self._load,
            on_patch=self._on_patch,
        )

        observer = Observer(timeout=self._poll_interval)
        watch_dir = self._path.rsplit("/", 1)[0]
        observer.schedule(handler, path=watch_dir, recursive=False)
        observer.start()

        self._observer = observer

    def stop(self) -> None:
        if self._observer is None:
            return

        self._observer.stop()
        self._observer.join()
        self._observer = None
	
