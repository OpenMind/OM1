import asyncio
import logging
import threading
import time
import typing as T
from concurrent.futures import ThreadPoolExecutor

from backgrounds.base import Background
from runtime.multi_mode.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages and coordinates background tasks for the application.

    Handles concurrent execution of multiple background tasks in separate
    threads, ensuring they run independently without blocking the main event loop.
    Supports graceful shutdown and error handling for individual background tasks.

    This class supports the context manager protocol for proper resource cleanup:
        with BackgroundOrchestrator(config) as orchestrator:
            orchestrator.start()
            # ... use orchestrator ...
    """

    _config: RuntimeConfig
    _background_workers: int
    _background_executor: ThreadPoolExecutor
    _submitted_backgrounds: set[str]
    _stop_event: threading.Event
    _stopped: bool

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the BackgroundOrchestrator with the provided configuration.

        Parameters
        ----------
        config : RuntimeConfig
            Configuration object for the runtime.
        """
        self._config = config
        self._background_workers = (
            min(12, len(config.backgrounds)) if config.backgrounds else 1
        )
        self._background_executor = ThreadPoolExecutor(
            max_workers=self._background_workers,
        )
        self._submitted_backgrounds = set()
        self._stop_event = threading.Event()
        self._stopped = False

    def start(self) -> asyncio.Future:
        """
        Start background tasks in separate threads.

        Submits each background task to the thread pool executor for concurrent
        execution. Skips backgrounds that have already been submitted to prevent
        duplicates.

        Returns
        -------
        asyncio.Future
            A future object for compatibility with async interfaces.
        """
        for background in self._config.backgrounds:
            if background.name in self._submitted_backgrounds:
                logging.warning(
                    f"Background {background.name} already submitted, skipping."
                )
                continue
            self._background_executor.submit(self._run_background_loop, background)
            self._submitted_backgrounds.add(background.name)

        return asyncio.Future()

    def _run_background_loop(self, background: Background):
        """
        Thread-based background loop.

        Parameters
        ----------
        background : Background
            The background task to run.
        """
        while not self._stop_event.is_set():
            try:
                background.run()
            except Exception as e:
                logging.error(f"Error in background {background.name}: {e}")
                time.sleep(0.1)

    def stop(self) -> None:
        """
        Stop the background executor and wait for all tasks to complete.

        Sets the stop event to signal all background loops to terminate,
        then shuts down the thread pool executor and waits for all running
        tasks to finish gracefully.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        self._background_executor.shutdown(wait=True)

    def __enter__(self) -> "BackgroundOrchestrator":
        """
        Enter the context manager.

        Returns
        -------
        BackgroundOrchestrator
            The orchestrator instance.
        """
        return self

    def __exit__(
        self,
        exc_type: T.Optional[type],
        exc_val: T.Optional[BaseException],
        exc_tb: T.Optional[T.Any],
    ) -> None:
        """
        Exit the context manager and ensure resources are cleaned up.

        Parameters
        ----------
        exc_type : Optional[type]
            The exception type if an exception was raised.
        exc_val : Optional[BaseException]
            The exception value if an exception was raised.
        exc_tb : Optional[Any]
            The traceback if an exception was raised.
        """
        self.stop()

    def __del__(self) -> None:
        """
        Clean up the BackgroundOrchestrator by stopping the executor.

        Note: This is a fallback cleanup mechanism. Prefer using the context
        manager protocol or explicitly calling stop() for reliable cleanup.
        """
        try:
            self.stop()
        except Exception:
            # Suppress exceptions during garbage collection
            pass
