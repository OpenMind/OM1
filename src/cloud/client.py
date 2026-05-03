"""WebSocket client for the OpenMind cloud broker.

Speaks the broker's binary frame format and JSON control protocol. Used
by `zenoh_shim.OpenMindZenohSession`; not called by OM1 plugins directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import threading
import uuid
from typing import Callable

import websockets

log = logging.getLogger("openmind.cloud.client")

PayloadCallback = Callable[[dict], None]
BinaryCallback = Callable[[bytes], None]


class CloudClient:
    """Synchronous-friendly client. The asyncio loop runs in a daemon thread."""

    def __init__(self, url: str, token: str | None = None, *, ca_path: str | None = None) -> None:
        # ca_path is for self-signed broker certs (test setups); falls back to
        # OPENMIND_CLOUD_VERIFY_TLS env var, then the system CA bundle.
        self._url = self._with_token(url, token) if token else url
        self._ssl_ctx = self._build_ssl_context(url, ca_path)
        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._subs: dict[str, PayloadCallback | BinaryCallback] = {}
        self._sub_binary: dict[str, bool] = {}
        # Remembered so we can replay them after a reconnect.
        self._sub_specs: dict[str, dict] = {}
        self._connected = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._connected.wait(timeout=10):
            raise RuntimeError(f"failed to connect to {url} within 10s")

    @staticmethod
    def _build_ssl_context(url: str, ca_path: str | None) -> ssl.SSLContext | None:
        if not url.lower().startswith("wss://"):
            return None
        ca = ca_path or os.environ.get("OPENMIND_CLOUD_VERIFY_TLS")
        return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()

    @staticmethod
    def _with_token(url: str, token: str) -> str:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}token={token}"

    # ----- public API -------------------------------------------------------

    def declare_subscriber(
        self,
        topic: str,
        schema: str,
        callback: PayloadCallback | BinaryCallback,
        *,
        binary: bool = False,
    ) -> str:
        """Subscribe to `topic` and return the sub_id.

        binary=True delivers raw CDR bytes (used for high-rate Image/PointCloud2
        to skip JSON encode/decode); binary=False delivers a decoded dict.
        """
        sub_id = uuid.uuid4().hex[:12]
        self._subs[sub_id] = callback
        self._sub_binary[sub_id] = binary
        self._sub_specs[sub_id] = {"topic": topic, "schema": schema, "binary": binary}
        self._send(
            {
                "type": "subscribe",
                "id": sub_id,
                "topic": topic,
                "schema": schema,
                "binary": binary,
            }
        )
        return sub_id

    def undeclare_subscriber(self, sub_id: str) -> None:
        """Cancel the subscription identified by `sub_id`."""
        self._subs.pop(sub_id, None)
        self._sub_binary.pop(sub_id, None)
        self._sub_specs.pop(sub_id, None)
        self._send({"type": "unsubscribe", "id": sub_id})

    def publish(self, topic: str, schema: str, payload: dict) -> None:
        """Publish a JSON payload; the server encodes it to CDR."""
        self._send({"type": "publish", "topic": topic, "schema": schema, "payload": payload})

    def publish_binary(self, topic: str, cdr_bytes: bytes) -> None:
        """Publish pre-encoded CDR bytes via the binary frame format.

        Frame: [0x20][2B topic len BE][topic UTF-8][CDR bytes]
        """
        topic_b = topic.encode("utf-8")
        if len(topic_b) > 65535:
            raise ValueError("topic too long")
        frame = bytes([0x20]) + len(topic_b).to_bytes(2, "big") + topic_b + cdr_bytes
        self._send_bytes(frame)

    def close(self) -> None:
        """Stop the reconnect loop and close the underlying websocket."""
        self._stopped.set()
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self._thread.join(timeout=5)

    # ----- internals --------------------------------------------------------

    def _send(self, msg: dict) -> None:
        if self._ws is None:
            raise RuntimeError("not connected")
        coro = self._ws.send(json.dumps(msg, separators=(",", ":")))
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _send_bytes(self, data: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("not connected")
        coro = self._ws.send(data)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        # Reconnect with exponential backoff so a transient broker drop
        # doesn't permanently stall the plugin's stream.
        connect_kwargs: dict = {
            # Camera frames can be multi-MB; default 1MB cap would drop them.
            "max_size": 32
            * 1024
            * 1024,
        }
        if self._ssl_ctx is not None:
            connect_kwargs["ssl"] = self._ssl_ctx

        backoff = 0.5
        while not self._stopped.is_set():
            try:
                async with websockets.connect(self._url, **connect_kwargs) as ws:
                    self._ws = ws
                    backoff = 0.5
                    # Server forgets subs on disconnect; replay them.
                    for sub_id, spec in list(self._sub_specs.items()):
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "subscribe",
                                    "id": sub_id,
                                    "topic": spec["topic"],
                                    "schema": spec["schema"],
                                    "binary": spec["binary"],
                                },
                                separators=(",", ":"),
                            )
                        )
                    self._connected.set()
                    async for raw in ws:
                        if self._stopped.is_set():
                            break
                        if isinstance(raw, (bytes, bytearray)):
                            self._dispatch_binary(bytes(raw))
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("server sent non-JSON text frame")
                            continue
                        self._dispatch_json(msg)
            except Exception as e:
                log.warning("cloud WS dropped: %s — reconnecting in %.1fs", e, backoff)
            finally:
                self._ws = None

            if self._stopped.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _dispatch_json(self, msg: dict) -> None:
        mt = msg.get("type")
        if mt == "sample":
            sub_id = msg.get("subId")
            cb = self._subs.get(sub_id)
            if cb is not None:
                try:
                    cb(msg.get("payload") or {})
                except Exception:
                    log.exception("subscriber callback raised")
        elif mt == "ack":
            log.debug("ack id=%s", msg.get("id"))
        elif mt == "error":
            log.warning("server error: %s", msg)
        elif mt == "pong":
            pass
        else:
            log.warning("unknown server message type: %s", mt)

    def _dispatch_binary(self, frame: bytes) -> None:
        # Frame: [0x10][2B subId len BE][subId UTF-8][CDR bytes]
        if len(frame) < 4 or frame[0] != 0x10:
            log.warning("server sent unknown binary frame type=%#x", frame[0] if frame else -1)
            return
        slen = int.from_bytes(frame[1:3], "big")
        if 3 + slen > len(frame):
            log.warning("malformed binary sample frame")
            return
        sub_id = frame[3 : 3 + slen].decode("utf-8", errors="replace")
        cdr = frame[3 + slen :]
        cb = self._subs.get(sub_id)
        if cb is None:
            return
        try:
            cb(cdr)
        except Exception:
            log.exception("binary subscriber callback raised")
