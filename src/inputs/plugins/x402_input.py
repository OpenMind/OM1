import asyncio
import logging
import threading
import time
from decimal import Decimal
from queue import Empty, Queue
from typing import Optional

from flask import Flask, jsonify, request
from x402 import x402ResourceServerSync
from x402.http.facilitator_client import HTTPFacilitatorClientSync
from x402.http.middleware.flask import payment_middleware
from x402.http.types import PaymentOption, RouteConfig

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class X402Input(FuserInput[SensorConfig, Optional[Message]]):
    """
    HTTP server input that gates incoming messages behind x402 payments.

    Exposes a Flask HTTP endpoint that requires x402 payment before
    accepting messages. Received messages are buffered and made
    available to the LLM through the standard input pipeline.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize the x402 input server.

        Parameters
        ----------
        config : SensorConfig
            Configuration for the x402 input.
        """
        super().__init__(config)

        self.descriptor_for_LLM = getattr(self.config, "input_name", "X402 Input")
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.message_buffer: Queue[Message] = Queue()

        self.fee = str(getattr(self.config, "fee", Decimal("0.01")))
        self.pay_to = getattr(self.config, "pay_to", None)
        self.network = getattr(self.config, "network", "base-sepolia")

        if self.pay_to is None:
            logging.warning(
                "x402 input: pay_to address not configured, "
                "payments will not be verified"
            )

        self.app = Flask(__name__)

        @self.app.route("/x402", methods=["POST"])
        def x402_handler():
            data = request.get_json()
            if not data or "message" not in data:
                return jsonify({"error": "No message provided"}), 400

            message = data["message"]
            timestamp = time.time()
            self.message_buffer.put(Message(timestamp=timestamp, message=message))
            return (
                jsonify({"status": "success", "timestamp": timestamp}),
                200,
            )

        if self.pay_to:
            payment_option = PaymentOption(
                scheme="exact",
                pay_to=self.pay_to,
                price=self.fee,
                network=self.network,
            )
            route_config = RouteConfig(
                accepts=payment_option,
                description="X402 Input - pay to send messages",
            )
            facilitator_client = HTTPFacilitatorClientSync()
            server = x402ResourceServerSync(
                facilitator_clients=[facilitator_client],
            )
            payment_middleware(
                app=self.app,
                routes={"/x402": route_config},
                server=server,
            )
            logging.info("x402 payment middleware configured")

        host = getattr(self.config, "host", "localhost")
        port = getattr(self.config, "port", 8765)

        self.flask_thread = threading.Thread(
            target=self._run_flask_app, args=(host, port), daemon=True
        )
        self.flask_thread.start()
        logging.info(f"x402 input server started on {host}:{port}")

    def _run_flask_app(self, host: str, port: int) -> None:
        """
        Run the Flask app in a background thread.

        Parameters
        ----------
        host : str
            The host address for the Flask app.
        port : int
            The port number for the Flask app.
        """
        self.app.run(host=host, port=port, debug=False, use_reloader=False)

    async def _poll(self) -> Optional[Message]:
        """
        Poll for new messages from the x402 HTTP server.

        Returns
        -------
        Optional[Message]
            The next message from the buffer if available, None otherwise.
        """
        await asyncio.sleep(0.5)
        try:
            return self.message_buffer.get_nowait()
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[Message]) -> Optional[Message]:
        """
        Pass through the already-formed Message object.

        Parameters
        ----------
        raw_input : Optional[Message]
            Message from the buffer.

        Returns
        -------
        Optional[Message]
            The same message, or None if input is None.
        """
        if raw_input is None:
            return None
        logging.debug(f"x402 input received: {raw_input.message}")
        return raw_input

    async def raw_to_text(self, raw_input: Optional[Message]) -> None:
        """
        Process raw input and add to the message buffer.

        Parameters
        ----------
        raw_input : Optional[Message]
            Raw input to be processed.
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message for the LLM prompt,
        registers it with IOProvider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            latest_message.message,
            latest_message.timestamp,
        )
        self.messages = []

        return result
