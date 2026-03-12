"""
Conversation history provider.

Polls IOProvider for new voice inputs in a background thread and
emits them via registered callbacks. Follows the same pattern as
FacePresenceProvider.

Place this file at: src/providers/conversation_history_provider.py
"""

import logging
import threading
import time
from typing import Callable, List, Optional

from .io_provider import IOProvider
from .singleton import singleton


@singleton
class ConversationHistoryProvider:
    """
    Singleton provider that polls IOProvider for voice inputs at a fixed cadence
    and emits text lines via registered callbacks.

    Parameters
    ----------
    max_rounds : int
        Maximum number of voice inputs to keep (default: 3).
    poll_interval : float
        Polling interval in seconds (default: 0.2).
    """

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        poll_interval: float = 0.2,
    ) -> None:
        self.max_rounds = max_rounds
        self.poll_interval = poll_interval
        self.io_provider = IOProvider()

        self._last_recorded_tick: int = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str], None]] = []
        self._cb_lock = threading.Lock()

        logging.info(
            f"ConversationHistoryProvider initialized: max_rounds={max_rounds}"
        )

    def register_message_callback(self, fn: Callable[[str], None]) -> None:
        """
        Subscribe a consumer to receive each emitted voice input line.

        Parameters
        ----------
        fn : Callable[[str], None]
            Function invoked from the polling thread with one voice input string.
        """
        with self._cb_lock:
            if fn not in self._callbacks:
                self._callbacks.append(fn)
                logging.info("Registered message callback")

    def unregister_message_callback(self, fn: Callable[[str], None]) -> None:
        """
        Remove a previously registered consumer.

        Parameters
        ----------
        fn : Callable[[str], None]
            The same callable passed to register_message_callback().
        """
        with self._cb_lock:
            try:
                self._callbacks.remove(fn)
            except ValueError:
                pass

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="conv-history-poll", daemon=True
        )
        self._thread.start()

    def stop(self, *, wait: bool = False) -> None:
        """
        Request the background thread to stop.

        Parameters
        ----------
        wait : bool
            If True, waits for the thread to finish. Defaults to False.
        """
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        """
        Internal polling loop.

        Checks IOProvider for new voice inputs and emits them via callbacks.
        """
        while not self._stop.is_set():
            try:
                text = self._fetch_voice_input()
                if text:
                    self._emit(text)
            except Exception as e:
                logging.warning(f"ConversationHistory poll error: {e}")
            time.sleep(self.poll_interval)

    def _emit(self, text: str) -> None:
        """
        Deliver one voice input line to all subscribers.

        Parameters
        ----------
        text : str
            The user's voice input text.
        """
        with self._cb_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(text)
            except Exception as e:
                logging.warning(f"ConversationHistory callback failed: {e}")

    def _fetch_voice_input(self) -> Optional[str]:
        """
        Check IOProvider for a new voice input this tick.

        Returns
        -------
        Optional[str]
            The voice input text if new, None otherwise.
        """
        current_tick = self.io_provider.tick_counter
        if current_tick <= self._last_recorded_tick:
            return None

        voice_input = self.io_provider.get_input("Voice")
        if voice_input and voice_input.input and voice_input.tick == current_tick:
            text = voice_input.input.strip()
            if text:
                self._last_recorded_tick = current_tick
                logging.debug(f"ConversationHistory: captured voice '{text[:50]}'")
                return text

        return None

    def clear(self) -> None:
        """Reset tick tracking (e.g., on mode transition)."""
        self._last_recorded_tick = -1
        logging.debug("ConversationHistory: cleared")
