import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backgrounds.base import Background
from runtime.multi_mode.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages the background tasks for the application.
    """

    _config: RuntimeConfig
    _background_workers: int
    _background_executor: ThreadPoolExecutor
    _submitted_backgrounds: set[str]
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
        self._submitted_backgrounds = set()
        self._stop_event = threading.Event()

    def start(self):
        """
        Start background tasks in separate threads.
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

    def stop(self):
        """
        Stop the background executor and wait for all tasks to complete.
        """
        self._stop_event.set()
        self._background_executor.shutdown(wait=True)

    def __del__(self):
        """
        Clean up the BackgroundOrchestrator by stopping the executor.
        """
        self.stop()
