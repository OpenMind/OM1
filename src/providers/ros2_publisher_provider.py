#!/usr/bin/env python3
import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from std_msgs.msg import String  # type: ignore

rclpy.init()


class ROS2PublisherProvider(Node):
    """
    Publisher provider for ROS 2.

    This class extends ROS 2 Node to provide a publisher that queues and
    publishes messages asynchronously in a separate thread. Messages are
    added to a queue and processed sequentially by a background thread.
    """

    def __init__(self, topic: str = "speak_topic"):
        """
        Initialize the ROS 2 Publisher Provider.

        Parameters
        ----------
        topic : str, optional
            The ROS 2 topic name to publish messages to. Defaults to
            "speak_topic". The publisher uses a queue size of 10.
        """
        try:
            super().__init__("ROS2_publisher_provider")
        except Exception as e:
            logging.error(f"Node initialization error: {e}")

        # Initialize the publisher.
        try:
            self.publisher_ = self.create_publisher(String, topic, 10)
            logging.info(f"Initialized ROS 2 publisher on topic '{topic}'")
        except Exception as e:
            logging.exception(f"Failed to create publisher on topic '{topic}': {e}")

        # Pending message queue and threading constructs
        self._pending_messages = Queue()
        self._lock = threading.Lock()
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None

    def add_pending_message(self, text: str):
        """
        Queue a message to be published.

        Parameters
        ----------
        text : str
            The text message to publish.
        """
        try:
            msg = String()
            # Append a timestamp to the message text.
            msg.data = f"{text} - {time.time()}"
            logging.info(f"Queueing message: {msg.data}")
            self._pending_messages.put(msg)
        except Exception as e:
            logging.exception(f"Error adding pending message: {e}")

    def _publish_message(self, msg: String):
        """
        Publish a single message and log the result.

        Parameters
        ----------
        msg : String
            The ROS 2 String message to publish.
        """
        try:
            self.publisher_.publish(msg)
            logging.info(f"Published message: {msg.data}")
        except Exception as e:
            logging.exception(f"Error publishing message: {e}")
        return None

    def start(self):
        """
        Start the publisher provider by launching the processing thread.
        """
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("ROS2 Publisher Provider started")

    def _run(self):
        """
        Internal loop that processes and publishes pending messages.
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

        self.publisher_.Close()

        logging.info("ROS2 Publisher Provider stopped")
