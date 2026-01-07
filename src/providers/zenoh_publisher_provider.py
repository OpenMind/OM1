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

    This class manages a Zenoh session, a message queue, and a worker thread that
    continuously publishes queued messages to a specified topic.
    """

    def __init__(self, topic: str = "speech"):
        """
        Initialize the Zenoh publisher provider and create a Zenoh session.

        Parameters
        ----------
        topic : str, optional
            The topic on which to publish messages (default is "speech").
        """
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self.session = None

        self.pub_topic = topic

        # Pending message queue and threading constructs
        self._pending_messages = Queue()
        self._lock = threading.Lock()
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None

    def add_pending_message(self, text: str):
        """
        Create a speech data message from the provided text and add it to the queue.

        Parameters
        ----------
        text : str
            The textual content to be published as a message.
        """
        msg = {"time_stamp": time.time(), "message": text}
        logging.info(f"Queueing message: {msg}")
        self._pending_messages.put(msg)

    def _publish_message(self, msg: dict):
        """
        Publish a single message using the Zenoh session.

        Parameters
        ----------
        msg : The message to be published.
        """
        # Attempt to write the message with a timeout of 0.5 seconds

        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return
        logging.info("Publishing message: {} ".format(msg))
        payload = ZBytes(json.dumps(msg))
        self.session.put(self.pub_topic, payload)

    def start(self):
        """
        Start the publisher provider by launching the processing thread.
        """
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("Zenoh Publisher Provider started")

    def _run(self):
        """
        Process and publish pending messages in an internal loop.
        """
        while self.running:
            try:
                # Wait up to 0.5 seconds for a message.
                msg = self._pending_messages.get(timeout=0.5)
                self._publish_message(msg)
            except Empty:
                continue
            except Exception as e:
                logging.exception("Exception in publisher thread: %s", e)

    def stop(self):
        """
        Stop the publisher provider and clean up resources.
        """
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.session is not None:
            self.session.close()
        logging.info("Zenoh Publisher Provider stopped")
