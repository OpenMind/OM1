import logging
from typing import Callable, Optional

import zenoh

from zenoh_msgs import open_zenoh_session

from .prometheus_monitor import PrometheusMonitor


class ZenohListenerProvider:
    """
    Listener provider for subscribing messages using a Zenoh session.

    This class manages a Zenoh session, a message queue, and a worker thread that
    continuously listens to messages to a specified topic.
    """

    def __init__(self, topic: str = "speech"):
        """
        Initialize the Zenoh Listener provider and create a Zenoh session.

        Parameters
        ----------
        topic : str, optional
            The topic on which to subscribe messages (default is "speech").
        """
        self.session: Optional[zenoh.Session] = None
        self.sub_topic = topic
        self.running: bool = False

        # Register with Prometheus monitor
        self._monitor = PrometheusMonitor()
        self._monitor.register(
            "ZenohListenerProvider",
            metadata={"type": "zenoh_listener", "topic": topic},
            recovery_callback=self._recover,
        )

        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            self._monitor.report_error("ZenohListenerProvider", str(e))

    def register_message_callback(self, message_callback: Optional[Callable]):
        """
        Register a callback function for processing incoming messages.

        Parameters
        ----------
        message_callback : Callable
            The function that will be called with each incoming Zenoh sample.
        """
        if self.session is not None and message_callback is not None:

            def wrapper(sample: zenoh.Sample) -> None:
                self._monitor.heartbeat("ZenohListenerProvider")
                message_callback(sample)

            self.session.declare_subscriber(self.sub_topic, wrapper)
        else:
            logging.error("Cannot register callback; Zenoh session is not available.")

    def start(self, message_callback: Optional[Callable] = None):
        """
        Start the listener provider by launching the background thread.
        """
        if self.running:
            logging.warning("Zenoh Listener Provider is already running")
            return

        if message_callback is not None:
            self.register_message_callback(message_callback)

        self.running = True
        logging.info("Zenoh Listener Provider started")
        self._monitor.heartbeat("ZenohListenerProvider")

    def stop(self):
        """
        Stop the listener provider and clean up resources.

        Stops the background thread and closes the Zenoh session.

        Notes
        -----
        The thread join operation uses a 5-second timeout to prevent hanging.
        """
        self.running = False

        if self.session is not None:
            self.session.close()

    def _recover(self) -> bool:
        """
        Attempt to recover the Zenoh listener provider.

        Returns
        -------
        bool
            True if recovery was successful, False otherwise.
        """
        try:
            logging.info("ZenohListenerProvider: Attempting recovery...")
            self.stop()
            self.start()
            logging.info("ZenohListenerProvider: Recovery successful")
            return True
        except Exception as e:
            logging.error(f"ZenohListenerProvider: Recovery failed: {e}")
            return False
