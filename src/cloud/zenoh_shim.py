"""Zenoh-session shim that lets unmodified OM1 plugins talk to the broker.

OM1 plugins call `open_zenoh_session()` and then use `declare_subscriber()`
/ `declare_publisher()` / `pub.put()` on a `zenoh.Session`. We return an
object with the same shape but route everything over the broker WS.

Each OM1 keyexpr is mapped to `(broker_topic, schema)` via a topic_map
so different products / robots can plug in their own conventions
without modifying OM1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .client import CloudClient

log = logging.getLogger("openmind.cloud.shim")


@dataclass(frozen=True)
class TopicSpec:
    """Mapping from an OM1 keyexpr to (broker_topic, ROS2 schema name)."""

    broker_topic: str
    schema: str


class _Payload:
    """Mimics zenoh.Sample.payload — has to_bytes() returning CDR."""

    __slots__ = ("_b",)

    def __init__(self, b: bytes) -> None:
        self._b = b

    def to_bytes(self) -> bytes:
        return self._b


class _Sample:
    """Mimics zenoh.Sample — only has the attrs OM1 plugins read."""

    __slots__ = ("key_expr", "payload")

    def __init__(self, key_expr: str, cdr: bytes) -> None:
        self.key_expr = key_expr
        self.payload = _Payload(cdr)


class _Subscriber:
    def __init__(self, session: "OpenMindZenohSession", key_expr: str, sub_id: str) -> None:
        self._session = session
        self._key_expr = key_expr
        self._sub_id = sub_id

    def undeclare(self) -> None:
        self._session._undeclare(self._sub_id)


class _NullSubscriber:
    """Returned for unmapped key_exprs — keeps OM1's internal IPC topics
    (om/mode/request, etc.) from crashing when they can't traverse the broker.
    """

    def __init__(self, key_expr: str) -> None:
        self._key_expr = key_expr

    def undeclare(self) -> None:
        pass


class _Publisher:
    def __init__(self, session: "OpenMindZenohSession", key_expr: str, spec: TopicSpec) -> None:
        self._session = session
        self._key_expr = key_expr
        self._spec = spec

    def put(self, payload: bytes) -> None:
        self._session._publish_binary(self._spec.broker_topic, payload)


class _NullPublisher:
    """No-op publisher for unmapped key_exprs — drops payloads silently."""

    def __init__(self, key_expr: str) -> None:
        self._key_expr = key_expr

    def put(self, payload: bytes) -> None:
        pass


class OpenMindZenohSession:
    """Drop-in replacement for `zenoh.Session` against the cloud broker."""

    def __init__(
        self,
        url: str,
        topic_map: dict[str, TopicSpec],
        *,
        token: str | None = None,
        strict: bool = False,
    ) -> None:
        # strict=False (default): unmapped key_exprs return no-op pub/sub.
        # strict=True: unmapped key_exprs raise KeyError. Useful in tests.
        self._client = CloudClient(url, token)
        self._topic_map = dict(topic_map)
        self._strict = strict
        self._subs: dict[str, str] = {}  # sub_id → key_expr

    def _resolve(self, key_expr: str) -> TopicSpec | None:
        spec = self._topic_map.get(key_expr)
        if spec is None and self._strict:
            raise KeyError(f"no topic mapping for key_expr={key_expr!r}")
        return spec

    # ----- public Zenoh-shaped API ------------------------------------------

    def declare_subscriber(
        self,
        key_expr: str,
        handler: Callable[[_Sample], None],
    ):
        """Subscribe a handler to messages on `key_expr` via the broker."""
        spec = self._resolve(key_expr)
        if spec is None:
            log.warning("subscribe to unmapped key_expr=%r — no-op", key_expr)
            return _NullSubscriber(key_expr)

        def _on_cdr(cdr: bytes, _ke=key_expr) -> None:
            try:
                handler(_Sample(_ke, cdr))
            except Exception:
                log.exception("OM1 handler raised on key=%s", _ke)

        sub_id = self._client.declare_subscriber(
            topic=spec.broker_topic,
            schema=spec.schema,
            callback=_on_cdr,
            binary=True,
        )
        self._subs[sub_id] = key_expr
        log.info("subscribed: OM1 key=%s → broker topic=%s (binary)", key_expr, spec.broker_topic)
        return _Subscriber(self, key_expr, sub_id)

    def declare_publisher(self, key_expr: str):
        """Return a publisher object whose `.put(bytes)` forwards to the broker."""
        spec = self._resolve(key_expr)
        if spec is None:
            log.warning("publisher on unmapped key_expr=%r — no-op", key_expr)
            return _NullPublisher(key_expr)
        log.info("publisher: OM1 key=%s → broker topic=%s", key_expr, spec.broker_topic)
        return _Publisher(self, key_expr, spec)

    def close(self) -> None:
        """Close the underlying broker connection."""
        self._client.close()

    # ----- internal helpers --------------------------------------------------

    def _undeclare(self, sub_id: str) -> None:
        self._subs.pop(sub_id, None)
        self._client.undeclare_subscriber(sub_id)

    def _publish_binary(self, broker_topic: str, cdr: bytes) -> None:
        self._client.publish_binary(broker_topic, cdr)
