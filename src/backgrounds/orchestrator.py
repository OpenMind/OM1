import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from backgrounds.base import Background
from runtime.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages and coordinates background tasks.

    Handles concurrent execution of multiple background tasks in separate
    threads, ensuring they run independently without blocking the main event loop.
    """

    _config: RuntimeConfig
    _background_workers: int
    _background_executor: ThreadPoolExecutor
    _submitted_backgrounds: set[str]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self._background_workers = (
            min(12, len(config.backgrounds)) if config.backgrounds else 1
        )
        self._background_executor = ThreadPoolExecutor(
            max_workers=self._background_workers,
        )
        self._submitted_backgrounds = set()
        self._stop_event = threading.Event()

    def start(self) -> asyncio.Future:
        """Start background tasks in separate threads."""
        for background in self._config.backgrounds:
            if background.name in self._submitted_backgrounds:
                logging.warning(
                    f"Background {background.name} already submitted, skipping."
                )
                continue

            background.set_stop_event(self._stop_event)
            self._background_executor.submit(self._run_background_loop, background)
            self._submitted_backgrounds.add(background.name)

        return asyncio.Future()

    def _run_background_loop(self, background: Background):
        while not self._stop_event.is_set():
            try:
                background.run()
            except Exception:
                logging.exception(f"Error in background {background.name}")
                self._stop_event.wait(timeout=0.1)

    def update_config(self, new_config: RuntimeConfig):
        """
        Update the internal config reference.
        Currently does not dynamically reconfigure running backgrounds.
        """
        self._config = new_config
        logging.info("BackgroundOrchestrator config reference updated.")

    def stop(self):
        """Stop background executor and wait for tasks to complete."""
        self._stop_event.set()
        self._background_executor.shutdown(wait=True)

    def __del__(self):
        """Clean up executor on deletion."""
        self.stop()
