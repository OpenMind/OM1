import logging
import threading
from typing import Callable, Optional

import zenoh

from zenoh_msgs import open_zenoh_session


class ZenohListenerProvider:
    """
    Listener provider for subscribing messages using a Zenoh session.
    """

    def __init__(self, topic: str = "speech"):
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self.session = None

        self.sub_topic = topic
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._subscriber = None
        self.running: bool = False

    def register_message_callback(self, message_callback: Callable):
        if self.session is None:
            logging.error("Cannot register callback; Zenoh session is not available.")
            return

        if self._subscriber is not None:
            logging.warning("Subscriber already registered")
            return

        self._subscriber = self.session.declare_subscriber(
            self.sub_topic, message_callback
        )

    def start(self, message_callback: Optional[Callable] = None):
        with self._lock:
            if self.running:
                logging.warning("Zenoh Listener Provider is already running")
                return

            if message_callback is not None:
                self.register_message_callback(message_callback)

            self._stop_event.clear()
            self.running = True
            logging.info("Zenoh Listener Provider started")

    def stop(self):
        with self._lock:
            if not self.running:
                return

            self._stop_event.set()
            self.running = False

            if self._subscriber is not None:
                try:
                    self._subscriber.undeclare()
                except Exception:
                    logging.exception("Error while undeclaring subscriber")
                self._subscriber = None

            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    logging.exception("Error while closing Zenoh session")

            logging.info("Zenoh Listener Provider stopped")
