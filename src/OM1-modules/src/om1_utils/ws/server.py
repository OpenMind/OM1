import asyncio
import logging
import threading
import uuid
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional, Union

from websockets import ConnectionClosed, WebSocketClientProtocol
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidHandshake, InvalidUpgrade

from ..healthcheck import HealthCheckServer

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

logging.getLogger("websockets.server").setLevel(logging.WARNING)


class Server:
    """
    An asynchronous WebSocket server implementation with support for multiple clients.

    This class provides a threaded WebSocket server that can handle multiple client
    connections, manage message queues for each connection, and support both targeted
    and broadcast message distribution.

    Parameters
    ----------
    host : str, optional
        The hostname to bind the server to, by default "localhost"
    port : int, optional
        The port number to listen on, by default 6789
    health_check_port : int, optional
        The port for the health check server, by default 8888
    """

    def __init__(
        self, host: str = "localhost", port: int = 6789, health_check_port: int = 8888
    ):
        self.host = host
        self.port = port
        self.running: bool = True
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.queues: Dict[str, Queue[str | bytes]] = {}
        # This queue is used to send messages to all connections
        self.global_queue: Queue[str | bytes] = Queue()
        self.connection_callback: Optional[Callable] = None
        self.message_callbacks: Dict[str, Optional[Callable]] = {}

        # Initialize health check server
        self.health_check = HealthCheckServer(port=health_check_port)

    def register_connection_callback(self, callback: Callable[[str, str], Any]):
        """
        Register a callback function for connection events.

        Parameters
        ----------
        callback : Callable[[str, str], Any]
            Function to be called on connection events. Takes event type
            ('connect'/'disconnect') and connection ID as arguments.
        """
        self.connection_callback = callback
        logger.info("Connection callback registered")

    def register_message_callback(self, connection_id: str, callback: Callable):
        """
        Register a callback function for a specific connection's messages.

        Parameters
        ----------
        connection_id : str
            The ID of the connection to register the callback for
        callback : Callable[[str, Union[str, bytes]], Any]
            Function to be called when messages are received on this connection
        """
        self.message_callbacks[connection_id] = callback
        logger.info(f"Registered message callback {connection_id}")

    async def process_global_messages(self):
        """
        Process messages in the global queue and send them to all connections.

        This coroutine continuously monitors the global message queue and broadcasts
        messages to all connected clients.
        """
        while self.running:
            try:
                message = self.global_queue.get_nowait()
                for connection_id in self.connections:
                    await self.connections[connection_id].send(message)
                self.global_queue.task_done()
            except Empty:
                await asyncio.sleep(0.05)
            except ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"Error sending global message: {e}")
                pass

    async def process_connection_messages(self, connection_id: str):
        """
        Process messages for a specific connection.

        Parameters
        ----------
        connection_id : str
            The ID of the connection to process messages for
        """
        while self.running and connection_id in self.connections:
            try:
                queue = self.queues[connection_id]
                message = queue.get_nowait()
                websocket = self.connections[connection_id]
                await websocket.send(message)
                queue.task_done()
            except Empty:
                await asyncio.sleep(0.05)
            except ConnectionClosed:
                break
            except Exception as e:
                logger.error(
                    f"Error sending message to connection {connection_id}: {e}"
                )
                break

    async def receive_messages(
        self, websocket: WebSocketClientProtocol, connection_id: str
    ):
        """
        Handle incoming messages for a specific connection.

        Parameters
        ----------
        websocket : WebSocketClientProtocol
            The WebSocket connection to receive messages from
        connection_id : str
            The ID of the connection
        """
        try:
            async for message in websocket:
                if connection_id in self.message_callbacks:
                    self.message_callbacks[connection_id](connection_id, message)
        except ConnectionClosed:
            logger.info(f"Connection closed: {connection_id}")
            pass
        except Exception as e:
            logger.error(
                f"Error receiving message from connection {connection_id}: {e}"
            )

    async def handle_connection(self, websocket: WebSocketClientProtocol):
        """
        Handle a new WebSocket connection.

        This coroutine manages the lifecycle of a WebSocket connection, including
        setup, message handling, and cleanup.

        Parameters
        ----------
        websocket : WebSocketClientProtocol
            The WebSocket connection to handle
        """
        connection_id = str(uuid.uuid4())
        try:
            # Store connection and create message queue
            self.connections[connection_id] = websocket
            self.queues[connection_id] = Queue()
            logger.info(f"New connection established: {connection_id}")

            # Notify about new connection if callback is registered
            if self.connection_callback:
                self.connection_callback("connect", connection_id)

            # Start tasks for sending and receiving messages
            send_global_task = asyncio.create_task(self.process_global_messages())
            send_task = asyncio.create_task(
                self.process_connection_messages(connection_id)
            )
            receive_task = asyncio.create_task(
                self.receive_messages(websocket, connection_id)
            )

            try:
                _, pending = await asyncio.wait(
                    [send_global_task, send_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

            except InvalidUpgrade as e:
                logger.warning(f"Invalid upgrade attempt: {e}")
                return
            except InvalidHandshake as e:
                logger.warning(f"Invalid handshake attempt: {e}")
                return
            except ConnectionClosed:
                logger.info(f"Connection closed normally: {connection_id}")
            except Exception as e:
                logger.error(
                    f"Unexpected error in connection handler: {e}", exc_info=True
                )

        finally:
            # Notify about connection closure if callback is registered
            if self.connection_callback:
                self.connection_callback("disconnect", connection_id)

            # Clean up connection resources
            if connection_id in self.queues:
                del self.queues[connection_id]
            if connection_id in self.connections:
                del self.connections[connection_id]
                logger.info(f"Connection closed: {connection_id}")

    def handle_response(self, connection_id: str, msg: Union[str, bytes]):
        """
        Queue a message to be sent to a specific connection.

        Parameters
        ----------
        connection_id : str
            The ID of the connection to send the message to
        msg : Union[str, bytes]
            The message to send
        """
        if connection_id in self.queues and self.running:
            self.queues[connection_id].put(msg)
            logger.info(
                f"Connection {connection_id} - Message queued: {self.format_message(msg)}"
            )

    def handle_global_response(self, msg: Union[str, bytes]):
        """
        Queue a message to be sent to all connected clients.

        Parameters
        ----------
        msg : Union[str, bytes]
            The message to broadcast to all clients
        """
        if self.running and self.connections:
            self.global_queue.put(msg)
            logger.info(f"Global message queued: {self.format_message(msg)}")

    async def start_server(self):
        """
        Start the WebSocket server and keep it running.
        """
        server = await serve(self.handle_connection, self.host, self.port)
        logger.info(f"WebSocket server started on port {self.port}")
        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            server.close()
            await server.wait_closed()

    def _run_server(self):
        """
        Internal method to run the server in a separate thread.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())

    def start(self):
        """
        Start the WebSocket server in a separate thread.
        """
        # Start health check server
        self.health_check.start()

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info("WebSocket server thread started")

    def format_message(self, msg: Union[str, bytes], max_length: int = 200) -> str:
        """
        Format a message for logging, truncating if necessary.

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

    def has_connections(self) -> bool:
        """
        Check if the server has any active connections.

        Returns
        -------
        bool
            True if there are active connections, False otherwise
        """
        return bool(self.connections)

    def stop(self):
        """
        Stop the WebSocket server.
        """
        self.running = False

        if self.health_check.is_running():
            self.health_check.stop()
