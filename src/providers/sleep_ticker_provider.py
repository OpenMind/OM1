import asyncio
import logging
import threading
from typing import Optional

from .singleton import singleton


@singleton
class SleepTickerProvider:
    """
    A singleton provider for managing asynchronous sleep operations with cancellation support.

    This class provides a thread-safe way to manage sleep operations that can be
    skipped/cancelled. It uses a lock mechanism to ensure thread safety when
    modifying the skip state.
    """

    def __init__(self):
        """
        Initialize the SleepTickerProvider with default values.

        The provider starts with sleep enabled (_skip_sleep = False) and no active
        sleep task.
        """
        self._lock: threading.Lock = threading.Lock()
        self._skip_sleep: bool = False
        self._current_sleep_task: Optional[asyncio.Task] = None

    @property
    def skip_sleep(self) -> bool:
        """
        Get the current skip sleep state.

        Returns
        -------
        bool
            True if sleep operations should be skipped, False otherwise.
        """
        with self._lock:
            return self._skip_sleep

    @skip_sleep.setter
    def skip_sleep(self, value: bool) -> None:
        """
        Set the skip sleep state and optionally cancel current sleep task.

        Parameters
        ----------
        value : bool
            The new skip sleep state. If True and there's an active sleep task,
            the task will be cancelled.
        """
        with self._lock:
            self._skip_sleep = value
            if value and self._current_sleep_task:
                self._current_sleep_task.cancel()

    async def sleep(self, duration: float) -> None:
        """
        Create and await an asynchronous sleep task.

        Creates a new sleep task and stores it as the current task. The task can
        be cancelled if skip_sleep is set to True while the sleep is in progress.

        Parameters
        ----------
        duration : float
            The duration to sleep in seconds.

        Returns
        -------
        None

        Raises
        ------
        asyncio.CancelledError
            If the sleep operation is cancelled, though this is caught internally.
        """
        if self.skip_sleep:
            return

        try:
            self._current_sleep_task = asyncio.create_task(asyncio.sleep(duration))
            await self._current_sleep_task
        except asyncio.CancelledError:
            logging.warning("Sleep operation was cancelled.")
            pass
        except Exception as e:
            logging.error(f"Error during sleep operation: {e}")
        finally:
            self._current_sleep_task = None
