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

    This class provides a thread-safe ROS 2 publisher that queues messages
    and publishes them asynchronously in a background thread. It handles
    message queuing, publishing, and thread lifecycle management.

    Attributes
    ----------
    publisher_ : rclpy.publisher.Publisher
        The ROS 2 publisher instance for String messages.
    _pending_messages : Queue
        Thread-safe queue for pending messages to be published.
    _lock : threading.Lock
        Lock for thread-safe operations.
    running : bool
        Flag indicating whether the publisher thread is running.
    _thread : Optional[threading.Thread]
        Background thread for message publishing.
    """

    def __init__(self, topic: str = "speak_topic"):
        """
        Initialize the ROS 2 publisher provider.

        Parameters
        ----------
        topic : str, optional
            The ROS 2 topic name to publish messages to. Defaults to "speak_topic".
            Must be a non-empty string.

        Raises
        ------
        ValueError
            If topic is None, empty string, or contains only whitespace.
        RuntimeError
            If ROS 2 node initialization fails or publisher creation fails.

        Notes
        -----
        The publisher is created with a queue size of 10. The node name
        is set to "ROS2_publisher_provider". Initialization errors are
        logged but do not prevent object creation.
        """
        if not topic or not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")

        try:
            super().__init__("ROS2_publisher_provider")
        except Exception as e:
            logging.error(f"Node initialization error: {e}")
            raise RuntimeError(f"Failed to initialize ROS 2 node: {e}") from e

        # Initialize the publisher.
        try:
            self.publisher_ = self.create_publisher(String, topic, 10)
            logging.info(f"Initialized ROS 2 publisher on topic '{topic}'")
        except Exception as e:
            logging.exception(f"Failed to create publisher on topic '{topic}': {e}")
            raise RuntimeError(f"Failed to create ROS 2 publisher: {e}") from e

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
            The text message to publish. Must be a non-empty string.

        Raises
        ------
        ValueError
            If text is None, empty string, or contains only whitespace.
        RuntimeError
            If message creation or queuing fails.

        Notes
        -----
        A timestamp is automatically appended to the message text before
        queuing. The message is added to the thread-safe queue and will
        be published by the background thread when it processes the queue.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            msg = String()
            # Append a timestamp to the message text.
            msg.data = f"{text} - {time.time()}"
            logging.info(f"Queueing message: {msg.data}")
            self._pending_messages.put(msg)
        except Exception as e:
            logging.exception(f"Error adding pending message: {e}")
            raise RuntimeError(f"Failed to queue message: {e}") from e

    def _publish_message(self, msg: String):
        """
        Publish a single message and log the result.

        Parameters
        ----------
        msg : String
            The ROS 2 String message to publish.

        Notes
        -----
        This method is called by the background thread to publish messages
        from the queue. Publishing errors are logged but do not stop the
        thread execution.
        """
        try:
            self.publisher_.publish(msg)
            logging.info(f"Published message: {msg.data}")
        except Exception as e:
            logging.exception(f"Error publishing message: {e}")

    def start(self):
        """
        Start the publisher provider by launching the processing thread.

        Notes
        -----
        This method is idempotent. If the thread is already running,
        calling this method has no effect. The background thread is
        created as a daemon thread, which will automatically terminate
        when the main program exits.
        """
        with self._lock:
            if self.running:
                logging.warning("ROS2 Publisher Provider is already running")
                return

            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logging.info("ROS2 Publisher Provider started")

    def _run(self):
        """
        Internal loop that processes and publishes pending messages.

        Notes
        -----
        This method runs in a background thread and continuously polls
        the message queue. It waits up to 0.5 seconds for each message.
        The loop terminates when `self.running` is set to False.
        Exceptions during message processing are logged but do not stop
        the thread execution.
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

        Notes
        -----
        This method sets the running flag to False, which causes the
        background thread to exit its loop. The method then waits up to
        5 seconds for the thread to complete. If the thread does not
        terminate within the timeout, the method continues execution.
        This method is idempotent and can be called multiple times safely.
        """
        with self._lock:
            if not self.running:
                logging.warning("ROS2 Publisher Provider is not running")
                return

            self.running = False

        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logging.warning("Publisher thread did not terminate within timeout")

        logging.info("ROS2 Publisher Provider stopped")
