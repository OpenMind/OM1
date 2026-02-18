import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from backgrounds.base import Background
from runtime.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages and coordinates background tasks for the application.

    Handles concurrent execution of multiple background tasks in separate
    threads, ensuring they run independently without blocking the main event loop.
    Supports graceful shutdown and error handling for individual background tasks.
    """

    _config: RuntimeConfig
    _background_workers: int
    _background_executor: ThreadPoolExecutor
    _background_instances: list[Background]
    _stop_event: threading.Event

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
        self._background_instances = []
        self._stop_event = threading.Event()

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
            if any(bg.name == background.name for bg in self._background_instances):
                logging.warning(
                    f"Background {background.name} already submitted, skipping."
                )
                continue

            background.set_stop_event(self._stop_event)

            self._background_executor.submit(self._run_background_loop, background)
            self._background_instances.append(background)

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
            except Exception:
                logging.exception(f"Error in background {background.name}")
                self._stop_event.wait(timeout=0.1)

    def stop(self):
        """
        Stop the background executor and wait for all tasks to complete.

        Sets the stop event to signal all background loops to terminate,
        calls stop() on each background instance for cleanup, then shuts
        down the thread pool executor and waits for all running tasks to
        finish gracefully.
        """
        self._stop_event.set()

        for background in self._background_instances:
            try:
                background.stop()
            except Exception as e:
                logging.error(f"Error stopping background {background.name}: {e}")

        self._background_executor.shutdown(wait=True)
        self._background_instances.clear()

    def __del__(self):
        """
        Clean up the BackgroundOrchestrator by stopping the executor.
        """
        self.stop()
