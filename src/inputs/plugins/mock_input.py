import asyncio
import logging
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

import websockets
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class MockSensorConfig(SensorConfig):
    """
    Configuration for Mock Sensor.

    Parameters
    ----------
    input_name : str
        Name of the input.
    host : str
        Host address for the WebSocket server.
    port : int
        Port number for the WebSocket server.
    """

    input_name: str = Field(default="Mock Input", description="Name of the input")
    host: str = Field(
        default="localhost", description="Host address for the WebSocket server"
    )
    port: int = Field(default=8765, description="Port number for the WebSocket server")


class MockInput(FuserInput[MockSensorConfig, Optional[str]]):
    """
    Mock Input Functionality.
    """

    def __init__(self, config: MockSensorConfig):
        """
        Initialize ASRInput instance.
        """
        super().__init__(config)

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Set IO Provider
        self.descriptor_for_LLM = self.config.input_name
        self.io_provider = IOProvider()

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # WebSocket configuration
        self.host = self.config.host
        self.port = self.config.port
        self.server = None
        self.connected_clients = set()
        self.loop = None

        # Start WebSocket server in a separate thread
        self.server_thread = threading.Thread(
            target=self._start_server_thread, daemon=True
        )
        self.server_thread.start()

    def _start_server_thread(self):
        """
        Start an asyncio event loop in a thread to run the websockets server.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())

        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    async def _start_server(self):
        """
        Start the WebSocket server.
        """
        try:
            self.server = await websockets.serve(
                self._handle_client, self.host, self.port
            )
            logging.info(
                f"Mock Input webSocket server started at ws://{self.host}:{self.port}"
            )
        except Exception as e:
            logging.error(f"Failed to start Mock Input webSocket server: {e}")

    async def _handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ):
        """
        Handle a client connection.

        Parameters
        ----------
        websocket : websockets.WebSocketServerProtocol
            The WebSocket connection
        path : str
            The connection path
        """
        self.connected_clients.add(websocket)
        logging.info(f"Client connected. Total clients: {len(self.connected_clients)}")

        try:
            async for message in websocket:
                try:
                    if isinstance(message, str):
                        text = message
                    else:
                        try:
                            text = message.decode("utf-8")
                        except Exception as e:
                            logging.warning(
                                f"Received binary data that couldn't be decoded. Skipping: {e}"
                            )
                            continue

                    logging.info(f"Received message: {text}")
                    self.message_buffer.put(text)

                    await websocket.send(f"Received: {text}")

                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                    await websocket.send(f"Error: {str(e)}")
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)

    async def _poll(self) -> Optional[str]:
        """
        Poll for new messages from the VLM service.

        Checks the message buffer for new messages with a brief delay
        to prevent excessive CPU usage.

        Returns
        -------
        Optional[str]
            The next message from the buffer if available, None otherwise
        """
        await asyncio.sleep(0.5)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1]}
// END
"""
        self.io_provider.add_input(
            self.descriptor_for_LLM, self.messages[-1].message, time.time()
        )
        self.messages = []
        return result
