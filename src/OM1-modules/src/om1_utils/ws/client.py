import logging
import threading
from queue import Empty, Queue
from typing import Callable, Optional, Union

import websockets
from websockets.sync.client import connect

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class Client:
    """
    A WebSocket client implementation with support for asynchronous message handling.

    This class provides a threaded WebSocket client that can maintain a persistent
    connection, automatically reconnect, and handle message sending and receiving
    asynchronously.

    Parameters
    ----------
    url : str, optional
        The WebSocket server URL to connect to, by default "ws://localhost:6789"
    """

    def __init__(self, url: str = "ws://localhost:6789"):
        self.url = url
        self.running: bool = True
        self.is_policy_violation: bool = False
        self.connected: bool = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.message_callback: Optional[Callable] = None
        self.message_queue: Queue = Queue()
        self.receiver_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None

    def _receive_messages(self):
        """
        Internal method to handle receiving messages from the WebSocket connection.

        Continuously receives messages and processes them through the registered callback
        if one exists. Runs in a separate thread.
        """
        while self.running and self.connected:
            try:
                message = self.websocket.recv()
                formatted_msg = self.format_message(message)
                logger.debug(f"Received WS Message: {formatted_msg}")
                if self.message_callback:
                    self.message_callback(message)
            except websockets.ConnectionClosed as e:
                close_code = e.code
                close_reason = e.reason
                if close_code == 1008:
                    self.is_policy_violation = True
                    logger.error("\n\n")
                    logger.error("----- Policy Violation -----")
                    logger.error(f"Policy violation: {close_reason}")
                    logger.error("----- Policy Violation -----\n\n")
                logger.info("WebSocket connection closed")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                self.connected = False
                break

    def _send_messages(self):
        """
        Internal method to handle sending messages through the WebSocket connection.

        Continuously processes messages from the message queue and sends them through
        the WebSocket connection. Runs in a separate thread.
        """
        while self.running:
            try:
                if self.connected and self.websocket:
                    message = self.message_queue.get_nowait()
                    try:
                        self.websocket.send(message)
                        formatted_msg = self.format_message(message)
                        logger.debug(f"Sent WS Message: {formatted_msg} to {self.url}")
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")
                        self.message_queue.put(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in send queue processing: {e}")
                self.connected = False

    def connect(self) -> bool:
        """
        Establish a connection to the WebSocket server.

        Attempts to connect to the WebSocket server and starts the receiver and sender
        threads if the connection is successful.

        Returns
        -------
        bool
            True if connection was successful, False otherwise
        """
        try:
            self.websocket = connect(self.url)
            self.connected = True

            # Start receiver and sender threads
            if not self.receiver_thread or not self.receiver_thread.is_alive():
                self.receiver_thread = threading.Thread(
                    target=self._receive_messages, daemon=True
                )
                self.receiver_thread.start()

            if not self.sender_thread or not self.sender_thread.is_alive():
                self.sender_thread = threading.Thread(
                    target=self._send_messages, daemon=True
                )
                self.sender_thread.start()

            logger.info(f"Connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False

    def send_message(self, message: str | bytes):
        """
        Queue a message to be sent through the WebSocket connection.

        Parameters
        ----------
        message : Union[str, bytes]
            The message to send, either as a string or bytes
        """
        if self.connected:
            self.message_queue.put(message)

    def _run_client(self):
        """
        Internal method to manage the WebSocket client lifecycle.

        Continuously attempts to maintain a connection to the WebSocket server,
        implementing automatic reconnection with a delay between attempts.
        """
        while self.running:
            if not self.connected and not self.is_policy_violation:
                if self.connect():
                    logger.info("Connection established")
                else:
                    logger.info("Connection failed, retrying in 5 seconds")
                    threading.Event().wait(5)  # Wait 5 seconds before retrying
            else:
                threading.Event().wait(1)

    def start(self):
        """
        Start the WebSocket client.

        Initializes and starts the main client thread that manages the WebSocket
        connection.
        """
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket client thread started")

    def register_message_callback(self, callback: Callable):
        """
        Register a callback function for handling received messages.

        Parameters
        ----------
        callback : Callable[[Union[str, bytes]], Any]
            Function to be called when a message is received. Should accept
            either string or bytes as input.
        """
        self.message_callback = callback
        logger.info("Registered message callback")

    def format_message(self, msg: Union[str, bytes], max_length: int = 200) -> str:
        """
        Format a message for logging purposes, truncating if necessary.

        Parameters
        ----------
        msg : Union[str, bytes]
            The message to format
        max_length : int, optional
            Maximum length of the formatted message, by default 200

        Returns
        -------
        str
            The formatted message string
        """
        try:
            if len(msg) <= max_length:
                return msg
            preview_size = max_length // 2 - 20
            return f"{msg[:preview_size]}...{msg[-preview_size:]}"
        except Exception as e:
            return f"<Error formatting message: {e}>"

    def is_connected(self) -> bool:
        """
        Check if the client is currently connected.

        Returns
        -------
        bool
            True if connected to the WebSocket server, False otherwise
        """
        return self.connected

    def stop(self):
        """
        Stop the WebSocket client.

        Closes the WebSocket connection, stops all threads, and cleans up resources.
        """
        self.running = False
        if self.websocket:
            try:
                self.websocket.close()
                logger.info("WebSocket connection closed")
            except Exception as _:
                pass

        try:
            while True:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
        except Empty:
            pass

        self.connected = False
        logger.info("WebSocket client stopped")
