import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional, Dict, Any

import zenoh
from zenoh import ZBytes

from zenoh_msgs import open_zenoh_session


class ZenohPublisherProvider:
    """
    Publisher provider for sending messages using a Zenoh session.
    """

    def __init__(self, topic: str = "speech", max_queue_size: int = 1000):
        self.pub_topic = topic
        self.running: bool = False

        self._pending_messages: Queue[Dict[str, Any]] = Queue(
            maxsize=max_queue_size
        )
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.session: Optional[zenoh.Session] = None

        self._open_session()

    def _open_session(self):
        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh session opened")
        except Exception as e:
            logging.exception("Failed to open Zenoh session: %s", e)
            self.session = None

    def add_pending_message(self, text: str) -> bool:
        msg = {
            "time_stamp": time.time(),
            "message": text,
        }

        try:
            self._pending_messages.put_nowait(msg)
            return True
        except Exception:
            logging.warning("Pending message queue is full, dropping message")
            return False

    def _publish_message(self, msg: Dict[str, Any]):
        if not self.session:
            logging.warning("No active Zenoh session, skipping publish")
            return

        try:
            payload = ZBytes(json.dumps(msg))
            self.session.put(self.pub_topic, payload)
            logging.debug("Published message to %s", self.pub_topic)
        except Exception as e:
            logging.exception("Failed to publish message: %s", e)

    def start(self):
        with self._lock:
            if self.running:
                logging.warning("Zenoh Publisher Provider already running")
                return

            self.running = True
            self._thread = threading.Thread(
                target=self._run, name="zenoh-publisher", daemon=True
            )
            self._thread.start()

        logging.info("Zenoh Publisher Provider started")

    def _run(self):
        while self.running:
            try:
                msg = self._pending_messages.get(timeout=0.5)
                self._publish_message(msg)
            except Empty:
                continue
            except Exception as e:
                logging.exception("Unhandled exception in publisher loop: %s", e)

    def stop(self):
        with self._lock:
            if not self.running:
                return
            self.running = False

        if self._thread:
            self._thread.join(timeout=5)

        if self.session:
            try:
                self.session.close()
                logging.info("Zenoh session closed")
            except Exception as e:
                logging.exception("Error while closing Zenoh session: %s", e)
            finally:
                self.session = None

        logging.info("Zenoh Publisher Provider stopped")
