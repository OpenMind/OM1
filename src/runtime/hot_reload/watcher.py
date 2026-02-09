import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_target_event(event: FileSystemEvent, target_path: str) -> bool:
    target = _normalize_path(target_path)
    src = getattr(event, "src_path", None)
    if isinstance(src, str) and src and _normalize_path(src) == target:
        return True
    dest = getattr(event, "dest_path", None)
    if isinstance(dest, str) and dest and _normalize_path(dest) == target:
        return True
    return False


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(
        self,
        target_path: str,
        on_change: Callable[[], None],
        debounce_seconds: float,
    ) -> None:
        super().__init__()
        self._target_path = os.path.abspath(target_path)
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._on_change)
            self._timer.daemon = True
            self._timer.start()

    def on_any_event(self, event: FileSystemEvent) -> None:
        try:
            if _is_target_event(event, self._target_path):
                self._schedule()
        except Exception:
            return

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


@dataclass
class AsyncFileWatcher:
    path: str
    loop: asyncio.AbstractEventLoop
    debounce_seconds: float = 0.25

    def __post_init__(self) -> None:
        self._event = asyncio.Event()
        self._observer: Optional[Observer] = None
        self._handler: Optional[_DebouncedHandler] = None

    def start(self) -> None:
        if self._observer is not None:
            return
        target_path = os.path.abspath(self.path)
        parent_dir = os.path.dirname(target_path)
        on_change = lambda: self.loop.call_soon_threadsafe(self._event.set)
        handler = _DebouncedHandler(
            target_path=target_path,
            on_change=on_change,
            debounce_seconds=self.debounce_seconds,
        )
        observer = Observer()
        observer.schedule(handler, parent_dir, recursive=False)
        observer.daemon = True
        observer.start()
        self._observer = observer
        self._handler = handler

    def stop(self) -> None:
        if self._handler is not None:
            self._handler.stop()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        self._observer = None
        self._handler = None

    async def wait_for_change(self) -> None:
        await self._event.wait()
        self._event.clear()
