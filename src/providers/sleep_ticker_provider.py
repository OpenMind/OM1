import asyncio
import threading
from typing import Optional

from .singleton import singleton


@singleton
class SleepTickerProvider:
    """
    A singleton provider for managing asynchronous sleep operations with
    cancellation and reset support.

    This provider ensures:
    - Thread-safe state updates
    - Proper async task lifecycle handling
    - Deterministic cancellation behavior
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._skip_sleep: bool = False
        self._current_sleep_task: Optional[asyncio.Task] = None

    @property
    def skip_sleep(self) -> bool:
        with self._lock:
            return self._skip_sleep

    @skip_sleep.setter
    def skip_sleep(self, value: bool) -> None:
        with self._lock:
            self._skip_sleep = value
            if value:
                self._cancel_current_task()

    def _cancel_current_task(self) -> None:
        task = self._current_sleep_task
        if task and not task.done():
            task.cancel()

    def reset(self) -> None:
        """
        Reset the sleep ticker state.

        Clears the skip flag and cancels any active sleep task.
        """
        with self._lock:
            self._skip_sleep = False
            self._cancel_current_task()

    async def sleep(self, duration: float) -> None:
        """
        Await an asynchronous sleep operation unless skipped.

        If skip_sleep is enabled, this method returns immediately.
        """
        if self.skip_sleep or duration <= 0:
            return

        task = asyncio.create_task(asyncio.sleep(duration))
        self._current_sleep_task = task

        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self._current_sleep_task is task:
                self._current_sleep_task = None
