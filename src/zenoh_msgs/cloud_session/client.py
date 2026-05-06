import asyncio
import json
import logging
import threading
import uuid
from typing import Callable, Optional, Union

import websockets

PayloadCallback = Callable[[dict], None]
BinaryCallback = Callable[[bytes], None]


class CloudZenohClient:
    """
    Websocket client for communicating with the cloud Zenoh broker.
    """

    def __init__(self, url: str, token: Optional[str] = None) -> None:
        """
        Initialize the CloudZenohClient.

        Parameters
        ----------
        url : str
            The URL of the cloud Zenoh broker to connect to.
        token : str, optional
            An optional authentication token to include in the connection URL.
        """
        self._url = self._with_token(url, token) if token else url
        self._websocket = None
        self._loop = asyncio.new_event_loop()
        self._subscriptions: dict[str, PayloadCallback | BinaryCallback] = {}
        self._subscription_is_binary: dict[str, bool] = {}

        self._subscription_specs: dict[str, dict] = {}
        self._connected = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _with_token(url: str, token: str) -> str:
        """
        Add the token as a query parameter to the URL, handling existing query parameters.

        Parameters
        ----------
        url : str
            The base URL to which the token should be added.
        token : str
            The token to be added as a query parameter.

        Returns
        -------
        str
            The URL with the token added as a query parameter.
        """
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}api_key={token}"

    def declare_subscriber(
        self,
        topic: str,
        callback: Union[PayloadCallback, BinaryCallback],
        binary: bool = False,
    ) -> str:
        """
        Subscribe to a topic with the given callback.

        Parameters
        ----------
        topic : str
            The topic to subscribe to.
        callback : Callable
            The callback function to be called when a message is received on the topic.
        binary : bool, optional
            Whether the subscription is for binary messages (CDR) instead of JSON payloads.

        Returns
        -------
        str
            A subscription ID that can be used to undeclare the subscriber.
        """
        sub_id = uuid.uuid4().hex[:12]
        self._subscriptions[sub_id] = callback
        self._subscription_is_binary[sub_id] = binary
        self._subscription_specs[sub_id] = {"topic": topic, "binary": binary}
        self._send(
            {
                "type": "subscribe",
                "id": sub_id,
                "topic": topic,
                "binary": binary,
            }
        )
        return sub_id

    def undeclare_subscriber(self, sub_id: str) -> None:
        """
        Cancel the subscription identified by `sub_id`.

        Parameters
        ----------
        sub_id : str
            The subscription ID returned by `declare_subscriber` to be cancelled.
        """
        self._subscriptions.pop(sub_id, None)
        self._subscription_is_binary.pop(sub_id, None)
        self._subscription_specs.pop(sub_id, None)
        self._send({"type": "unsubscribe", "id": sub_id})

    def publish(self, topic: str, payload: dict) -> None:
        """
        Publish a JSON payload; the server encodes it to CDR.

        Parameters
        ----------
        topic : str
            The topic to publish to.
        payload : dict
            The JSON-serializable payload to be published.
        """
        self._send({"type": "publish", "topic": topic, "payload": payload})

    def publish_binary(self, topic: str, encoded_message_bytes: bytes) -> None:
        """
        Publish pre-encoded CDR bytes via the binary frame format.

        Frame: [0x20][2B topic len BE][topic UTF-8][CDR bytes]

        Parameters
        ----------
        topic : str
            The topic to publish to.
        encoded_message_bytes : bytes
            The CDR-encoded message bytes to be published.
        """
        topic_b = topic.encode("utf-8")
        if len(topic_b) > 65535:
            raise ValueError("topic too long")
        frame = bytes([0x20]) + len(topic_b).to_bytes(2, "big") + topic_b + encoded_message_bytes
        self._send_bytes(frame)

    def close(self) -> None:
        """
        Stop the reconnect loop and close the underlying websocket.
        """
        self._stopped.set()
        if self._websocket is not None:
            asyncio.run_coroutine_threadsafe(self._websocket.close(), self._loop)
        self._thread.join(timeout=5)

    def _send(self, msg: dict) -> None:
        """
        Send a JSON message to the server, ensuring it's run in the event loop.

        Parameters
        ----------
        msg : dict
            The message to be sent to the server, which will be JSON-encoded.
        """
        if self._websocket is None:
            return

        coro = self._websocket.send(json.dumps(msg, separators=(",", ":")))
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _send_bytes(self, data: bytes) -> None:
        """
        Send raw bytes to the server, ensuring it's run in the event loop.

        Parameters
        ----------
        data : bytes
            The raw bytes to be sent to the server.
        """
        # Wait for the connection to be established (with a reasonable timeout)
        if not self._connected.wait(timeout=30.0):
            raise RuntimeError("not connected: timeout waiting for connection")
        if self._websocket is None:
            raise RuntimeError("not connected")
        coro = self._websocket.send(data)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run(self) -> None:
        """
        Run the websocket client in a separate thread, managing reconnections and message dispatch.
        """
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        """
        Main loop for managing the websocket connection, including automatic reconnection with exponential backoff.
        """
        connect_kwargs: dict = {
            # Camera frames can be multi-MB; default 1MB cap would drop them.
            "max_size": 32
            * 1024
            * 1024,
        }

        backoff = 0.5
        while not self._stopped.is_set():
            try:
                async with websockets.connect(self._url, **connect_kwargs) as websocket:
                    self._websocket = websocket
                    backoff = 0.5
                    for sub_id, spec in list(self._subscription_specs.items()):
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "subscribe",
                                    "id": sub_id,
                                    "topic": spec["topic"],
                                    "binary": spec["binary"],
                                },
                                separators=(",", ":"),
                            )
                        )
                    self._connected.set()

                    async for raw_message in websocket:
                        if self._stopped.is_set():
                            break

                        if isinstance(raw_message, (bytes, bytearray)):
                            self._dispatch_binary(bytes(raw_message))
                            continue

                        try:
                            msg = json.loads(raw_message)
                        except json.JSONDecodeError:
                            logging.warning("server sent non-JSON text frame")
                            continue

                        self._dispatch_json(msg)

            except Exception as e:
                logging.warning("cloud WS dropped: %s — reconnecting in %.1fs", e, backoff)
            finally:
                self._websocket = None

            if self._stopped.is_set():
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _dispatch_json(self, msg: dict) -> None:
        """
        Dispatch a JSON message from the server, routing it to the appropriate subscriber callback based on the message type.

        Parameters
        ----------
        msg : dict
            The JSON-decoded message received from the server, which should contain a "type" field indicating the message type and other relevant fields depending on the type.
        """
        message_type = msg.get("type")

        if message_type == "sample":
            sub_id = msg.get("subId")

            if sub_id is not None:
                callback = self._subscriptions.get(sub_id)
            else:
                callback = None

            if callback is not None:
                try:
                    callback(msg.get("payload") or {})  # type: ignore
                except Exception:
                    logging.exception("subscriber callback raised")

        elif message_type == "ack":
            logging.debug("ack id=%s", msg.get("id"))
        elif message_type == "error":
            logging.warning("server error: %s", msg)
        elif message_type == "pong":
            pass
        else:
            logging.warning("unknown server message type: %s", message_type)

    def _dispatch_binary(self, frame: bytes) -> None:
        """
        Dispatch a binary frame from the server, which is expected to be in the format.

        Payload frame: [0x10][2B subId len BE][subId UTF-8][CDR bytes]

        Parameters
        ----------
        frame : bytes
            The raw binary frame received from the server, which should follow the specified format.
        """
        if len(frame) < 4 or frame[0] != 0x10:
            logging.warning("server sent unknown binary frame type=%#x", frame[0] if frame else -1)
            return

        sub_id_length = int.from_bytes(frame[1:3], "big")
        if 3 + sub_id_length > len(frame):
            logging.warning("malformed binary sample frame")
            return

        sub_id = frame[3 : 3 + sub_id_length].decode("utf-8", errors="replace")
        encoded_message_data = frame[3 + sub_id_length :]
        callback = self._subscriptions.get(sub_id)

        if callback is None:
            return

        try:
            callback(encoded_message_data)  # type: ignore
        except Exception:
            logging.exception("binary subscriber callback raised")
