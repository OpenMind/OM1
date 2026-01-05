import json
import logging
import threading
import time
from typing import Any, Dict, Optional

import zenoh

from zenoh_msgs import open_zenoh_session
from .singleton import singleton


@singleton
class ContextProvider:
    """
    Singleton provider for managing and publishing mode-aware context via Zenoh.
    """

    def __init__(self):
        self.context_update_topic = "om/mode/context"
        self.session: Optional[zenoh.Session] = None
        self.publisher: Optional[Any] = None

        self._lock = threading.Lock()
        self._context: Dict[str, Any] = {}
        self._last_updated: Optional[float] = None

        self._initialize_zenoh()

    def _initialize_zenoh(self):
        try:
            session = open_zenoh_session()
            if session is None:
                raise RuntimeError("Zenoh session is None")

            self.session = session
            self.publisher = session.declare_publisher(self.context_update_topic)
            logging.info("ContextProvider Zenoh session initialized")

        except Exception as e:
            logging.error(f"ContextProvider Zenoh initialization failed: {e}")
            self.session = None
            self.publisher = None

    def _ensure_initialized(self) -> bool:
        if self.publisher is not None:
            return True

        logging.warning("ContextProvider not initialized, retrying Zenoh setup")
        self._initialize_zenoh()
        return self.publisher is not None

    def update_context(self, context: Dict[str, Any]):
        if not isinstance(context, dict):
            logging.error("Context update rejected: context must be a dict")
            return

        if not self._ensure_initialized():
            logging.error("Context update failed: Zenoh not available")
            return

        with self._lock:
            self._context.update(context)
            self._last_updated = time.time()
            payload = {
                "context": self._context,
                "timestamp": self._last_updated,
            }

        if self.publisher is None:
         logging.error("Context update skipped: publisher not available")
         return

        try:
           context_json = json.dumps(payload)
           self.publisher.put(context_json.encode("utf-8"))
           logging.debug(f"Context updated: {context}")

        except Exception as e:
            logging.error(f"Failed to publish context update: {e}")

    def set_context_field(self, key: str, value: Any):
        self.update_context({key: value})

    def get_context(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._context)

    def get_context_field(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._context.get(key, default)

    def clear_context(self):
        with self._lock:
            self._context.clear()
            self._last_updated = time.time()

        if not self._ensure_initialized():
            return

        if self.publisher is None:
         logging.error("Context clear skipped: publisher not available")
         return

        try:
            payload = {
            "context": {},
            "timestamp": self._last_updated,
                }
            self.publisher.put(json.dumps(payload).encode("utf-8"))
            logging.info("Context cleared")


        except Exception as e:
            logging.error(f"Failed to publish context clear event: {e}")

    def stop(self):
        with self._lock:
            self._context.clear()

        if self.session:
            try:
                self.session.close()
                logging.info("ContextProvider stopped")
            except Exception as e:
                logging.error(f"Error stopping ContextProvider: {e}")
            finally:
                self.session = None
                self.publisher = None