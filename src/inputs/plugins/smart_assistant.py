"""
Smart Assistant Input Plugin for OM1.

This plugin provides HTTP webhook integration for smart assistants
like Home Assistant, Alexa, and Siri. It listens for incoming
commands and forwards them to the OM1 agent.

Supported Features:
- HTTP webhook endpoint for receiving commands
- Home Assistant webhook integration
- Alexa Smart Home skill support
- IFTTT webhook compatibility
"""

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class SmartAssistantConfig(SensorConfig):
    """
    Configuration for Smart Assistant Input.

    Parameters
    ----------
    host : str
        Host address for the HTTP webhook server.
    port : int
        Port number for the HTTP webhook server.
    webhook_path : str
        URL path for the webhook endpoint.
    """

    host: str = Field(
        default="0.0.0.0",
        description="Host address for the HTTP webhook server",
    )
    port: int = Field(
        default=8080,
        description="Port number for the HTTP webhook server",
    )
    webhook_path: str = Field(
        default="/webhook",
        description="URL path for the webhook endpoint",
    )


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for smart assistant webhooks."""

    message_queue: Queue = Queue()
    webhook_path: str = "/webhook"

    def log_message(self, format: str, *args) -> None:
        """Override to use logging instead of stderr."""
        logging.debug(f"Webhook: {format % args}")

    def _send_response(self, status: int, data: Dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self._send_response(200, {"status": "ok"})

    def do_GET(self) -> None:
        """Handle GET requests - health check."""
        if self.path == "/health":
            self._send_response(
                200, {"status": "healthy", "service": "smart-assistant"}
            )
        else:
            self._send_response(
                200,
                {
                    "status": "ok",
                    "message": "Smart Assistant Webhook Server",
                    "endpoints": {
                        "webhook": self.webhook_path,
                        "health": "/health",
                    },
                },
            )

    def do_POST(self) -> None:
        """Handle POST requests - webhook commands."""
        if self.path != self.webhook_path:
            self._send_response(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")

            # Parse the request body
            try:
                data = json.loads(post_data) if post_data else {}
            except json.JSONDecodeError:
                # If not JSON, treat as plain text command
                data = {"command": post_data}

            # Extract command from various smart assistant formats
            command = self._extract_command(data)

            if command:
                WebhookHandler.message_queue.put(command)
                logging.info(f"Smart Assistant command received: {command}")
                self._send_response(
                    200,
                    {
                        "status": "received",
                        "command": command,
                    },
                )
            else:
                self._send_response(400, {"error": "No command found in request"})

        except Exception as e:
            logging.error(f"Error processing webhook: {e}")
            self._send_response(500, {"error": str(e)})

    def _extract_command(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract command from various smart assistant payload formats.

        Supports:
        - Home Assistant: {"action": "...", "data": {...}}
        - Alexa: {"request": {"intent": {...}}}
        - IFTTT: {"value1": "..."}
        - Generic: {"command": "..."}

        Parameters
        ----------
        data : Dict[str, Any]
            The incoming webhook payload

        Returns
        -------
        Optional[str]
            The extracted command, or None if not found
        """
        # Direct command field
        if "command" in data:
            return str(data["command"])

        # Home Assistant format
        if "action" in data:
            action = data["action"]
            action_data = data.get("data", {})
            if isinstance(action_data, dict):
                # Combine action with any relevant data
                amount = action_data.get("amount", "")
                recipient = action_data.get("recipient", "")
                if amount and recipient:
                    return f"{action} {amount} to {recipient}"
            return str(action)

        # Alexa format
        if "request" in data and "intent" in data.get("request", {}):
            intent = data["request"]["intent"]
            intent_name = intent.get("name", "")
            slots = intent.get("slots", {})
            # Build command from intent and slots
            command_parts = [intent_name]
            for slot_name, slot_data in slots.items():
                if "value" in slot_data:
                    command_parts.append(f"{slot_name}={slot_data['value']}")
            return " ".join(command_parts)

        # IFTTT format
        if "value1" in data:
            parts = [data.get("value1", "")]
            if "value2" in data:
                parts.append(data.get("value2", ""))
            if "value3" in data:
                parts.append(data.get("value3", ""))
            return " ".join(filter(None, parts))

        # Message field (generic)
        if "message" in data:
            return str(data["message"])

        # Text field
        if "text" in data:
            return str(data["text"])

        return None


class SmartAssistantInput(FuserInput[SmartAssistantConfig, Optional[str]]):
    """
    Smart Assistant Input for OM1.

    Provides HTTP webhook integration for smart home assistants
    like Home Assistant, Alexa, and Siri. Commands received via
    webhook are processed and forwarded to the OM1 agent.

    Example webhook payload for Home Assistant:
    ```json
    {
        "action": "pay",
        "data": {
            "amount": "0.001",
            "recipient": "0x1234..."
        }
    }
    ```

    Example webhook payload for Alexa:
    ```json
    {
        "request": {
            "intent": {
                "name": "PaymentIntent",
                "slots": {
                    "amount": {"value": "0.001"},
                    "recipient": {"value": "friend"}
                }
            }
        }
    }
    ```
    """

    def __init__(self, config: SmartAssistantConfig):
        """Initialize SmartAssistantInput instance."""
        super().__init__(config)

        self.messages: List[Message] = []
        self.io_provider = IOProvider()

        # Configure webhook handler
        WebhookHandler.webhook_path = self.config.webhook_path
        WebhookHandler.message_queue = Queue()
        self.message_queue = WebhookHandler.message_queue

        # HTTP server
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

        # Start the server
        self._start_server()

    def _start_server(self) -> None:
        """Start the HTTP webhook server in a background thread."""
        try:
            self.server = HTTPServer(
                (self.config.host, self.config.port),
                WebhookHandler,
            )
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
            )
            self.server_thread.start()
            logging.info(
                f"Smart Assistant webhook server started at "
                f"http://{self.config.host}:{self.config.port}{self.config.webhook_path}"
            )
        except Exception as e:
            logging.error(f"Failed to start Smart Assistant server: {e}")

    def _run_server(self) -> None:
        """Run the HTTP server."""
        if self.server:
            self.server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP webhook server."""
        if self.server:
            self.server.shutdown()
            logging.info("Smart Assistant webhook server stopped")

    async def _poll(self) -> Optional[str]:
        """
        Poll for new commands from smart assistants.

        Returns
        -------
        Optional[str]
            The next command from the queue if available, None otherwise
        """
        await asyncio.sleep(0.5)
        try:
            command = self.message_queue.get_nowait()
            return command
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Convert raw command to timestamped message.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw command string from smart assistant

        Returns
        -------
        Optional[Message]
            Timestamped message containing the command
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        """
        Process raw input and update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw command from smart assistant
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and return the latest command from buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the command, or None if empty
        """
        if len(self.messages) == 0:
            return None

        last_message = self.messages[-1]

        result = f"""
SmartAssistantInput INPUT
// START
Smart Assistant Command: {last_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__,
            last_message.message,
            last_message.timestamp,
        )
        self.messages = []
        return result

    def __del__(self) -> None:
        """Clean up server on destruction."""
        self.stop()
