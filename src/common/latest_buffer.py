# common/latest_buffer.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LatestItem(Generic[T]):
    """
    Value + timestamp container returned by LatestBuffer operations.
    """

    ts: float
    value: T


class LatestBuffer(Generic[T]):
    """
    A single-slot, 'latest only' buffer.

    - Thread-safe.
    - Fast path for producers: push() overwrites the slot and wakes waiters.
    - Consumers can:
        * peek_latest(): read without clearing
        * drain_latest(): read and clear the slot (so you won't re-read it)
        * wait_next(after_ts): block until a newer item arrives (or timeout)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[LatestItem[T]] = None

    def push(self, value: T) -> None:
        """
        Publish a new value (overwrites any previous one).
        Wakes up threads waiting in wait_next().
        """
        with self._cond:
            self._latest = LatestItem(ts=time.time(), value=value)
            self._cond.notify_all()

    def peek_latest(self) -> Optional[LatestItem[T]]:
        """
        Read the current value without clearing it.
        Returns None if the slot is empty.
        """
        with self._lock:
            return self._latest

    def drain_latest(self) -> Optional[LatestItem[T]]:
        """
        Read and clear the current value.
        This is useful when you want to ensure you don't process the same item twice.
        """
        with self._lock:
            item = self._latest
            if item is not None:
                self._latest = None
            return item

    def wait_next(
        self, after_ts: float, timeout: Optional[float] = None
    ) -> Optional[LatestItem[T]]:
        """
        Block until an item strictly newer than 'after_ts' is available, or until timeout.

        Parameters
        ----------
        after_ts : float
            Only return when a value with ts > after_ts exists.
        timeout : float | None
            Max seconds to wait. None means wait indefinitely.

        Returns
        -------
        LatestItem[T] | None
            The newer item, or None on timeout.
        """
        deadline = None if timeout is None else (time.time() + timeout)
        with self._cond:
            while True:
                if self._latest is not None and self._latest.ts > after_ts:
                    return self._latest
                remaining = None if deadline is None else (deadline - time.time())
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
