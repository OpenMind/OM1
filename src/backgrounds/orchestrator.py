import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from backgrounds.base import Background

if TYPE_CHECKING:
    from runtime.multi_mode.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages the background tasks for the application.
    """

    _config: "RuntimeConfig"
    _background_workers: int
    _background_executor: ThreadPoolExecutor
    _submitted_backgrounds: set[str]
    _background_futures: dict[str, Future]
    _stop_event: threading.Event

    def __init__(self, config: "RuntimeConfig"):
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
        self._background_futures = {}
        self._stop_event = threading.Event()

    def start(self) -> dict[str, Future]:
        """
        Start background tasks in separate threads.

        Returns
        -------
        dict[str, Future]
            A mapping between background names and the futures tracking their
            execution.
        """
        for background in self._config.backgrounds:
            if background.name in self._submitted_backgrounds:
                logging.warning(
                    f"Background {background.name} already submitted, skipping."
                )
                continue
            future = self._background_executor.submit(
                self._run_background_loop, background
            )
            self._background_futures[background.name] = future
            self._submitted_backgrounds.add(background.name)

        return dict(self._background_futures)

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
