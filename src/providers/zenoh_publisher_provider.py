import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

import zenoh
from zenoh import ZBytes

from zenoh_msgs import open_zenoh_session


class ZenohPublisherProvider:
    """
    Publisher provider for sending messages using a Zenoh session.
    """

    def __init__(self, topic: str = "speech", queue_size: int = 100):
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self.session = None

        self.pub_topic = topic

        self._pending_messages: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_pending_message(self, text: str) -> None:
        msg = {"time_stamp": time.time(), "message": text}

        if self._stop_event.is_set():
            logging.warning("Publisher stopped, dropping message")
            return

        try:
            self._pending_messages.put(msg, block=False)
            logging.info(f"Queueing message: {msg}")
        except Exception:
            logging.warning("Message queue full, dropping message")

    def _publish_message(self, msg: dict) -> None:
        if self.session is None:
            logging.warning("No active Zenoh session, dropping message")
            return

        payload = ZBytes(json.dumps(msg))
        self.session.put(self.pub_topic, payload)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logging.info("Zenoh Publisher Provider started")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._pending_messages.get(timeout=0.5)
                self._publish_message(msg)
            except Empty:
                continue
            except Exception:
                logging.exception("Unhandled exception in publisher thread")

        # Drain remaining messages on shutdown
        while not self._pending_messages.empty():
            try:
                msg = self._pending_messages.get_nowait()
                self._publish_message(msg)
            except Exception:
                break

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()

            if self._thread:
                self._thread.join(timeout=5)
                self._thread = None

            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    logging.exception("Error while closing Zenoh session")

            logging.info("Zenoh Publisher Provider stopped")
