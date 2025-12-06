import json
import logging
from typing import Any, Dict, Optional

import zenoh

from zenoh_msgs import open_zenoh_session

from .singleton import singleton


@singleton
class ContextProvider:
    """
    Singleton provider for updating mode-aware context via Zenoh messaging.

    This provider allows any component (inputs, actions, backgrounds) to update
    the user context that drives context-aware mode transitions.
    """

    def __init__(self):
        """
        Initialize the ContextProvider.
        """
        self.context_update_topic = "om/mode/context"
        self.session: Optional[zenoh.Session] = None
        self.publisher = None
        self._initialize_zenoh()

    def _initialize_zenoh(self):
        """
        Initialize Zenoh session and publisher.
        """
        try:
            self.session = open_zenoh_session()
            self.publisher = self.session.declare_publisher(self.context_update_topic)
            logging.info("ContextProvider Zenoh session initialized")
        except Exception as e:
            logging.error(f"Error initializing ContextProvider Zenoh session: {e}")
            self.session = None
            self.publisher = None

    def update_context(self, context: Dict[str, Any]):
        """
        Update the user context for context-aware transitions.

        Parameters
        ----------
        context : Dict[str, Any]
            The context information to update. This will be merged with existing context.
        """
        if not self.publisher:
            logging.warning("ContextProvider not initialized, cannot update context")
            return

        try:
            context_json = json.dumps(context)
            self.publisher.put(context_json.encode("utf-8"))
            logging.debug(f"Sent context update: {context}")
        except Exception as e:
            logging.error(f"Error sending context update: {e}")

    def set_context_field(self, key: str, value: Any):
        """
        Update a single context field.

        Parameters
        ----------
        key : str
            The context key to update
        value : Any
            The value to set for this key
        """
        self.update_context({key: value})

    def stop(self):
        """
        Stop the ContextProvider and clean up resources.
        """
        if self.session:
            try:
                self.session.close()
                logging.info("ContextProvider stopped")
            except Exception as e:
                logging.error(f"Error stopping ContextProvider: {e}")
            finally:
                self.session = None
                self.publisher = None
