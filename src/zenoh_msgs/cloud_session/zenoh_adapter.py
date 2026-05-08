import logging
from typing import Any, Callable, Optional, Union

from zenoh import ZBytes

from zenoh_msgs.cloud_session.client import CloudZenohClient


class _Payload:
    """
    Wraps the raw bytes payload from the broker, mimicking the Zenoh Sample's
    payload API.
    """

    def __init__(self, b: bytes) -> None:
        """
        Parameters
        ----------
        b : bytes
            The raw bytes payload from the broker.
        """
        self._b = b

    def to_bytes(self) -> bytes:
        """
        Return the raw bytes payload.

        Returns
        -------
        bytes
            The raw bytes payload.
        """
        return self._b


class _Sample:
    """
    Wraps a message sample from the broker, mimicking the Zenoh Sample API.
    """

    def __init__(self, topic: str, cdr: bytes) -> None:
        """
        Initialize a sample with the topic and raw bytes payload.

        Parameters
        ----------
        topic : str
            The topic this sample was published to.
        cdr : bytes
            The raw bytes payload from the broker.
        """
        self.topic = topic
        self.payload = _Payload(cdr)


class _Subscriber:
    """
    Represents an active subscription to a broker topic, mimicking the Zenoh Subscriber API.
    """

    def __init__(self, session: "CloudSimZenohSession", topic: str, sub_id: str) -> None:
        """
        Initialize a subscriber with the session, topic, and subscription ID.

        Parameters
        ----------
        session : CloudSimZenohSession
            The Zenoh session this subscriber belongs to.
        topic : str
            The topic this subscriber is subscribed to.
        sub_id : str
            The unique subscription ID returned by the broker.
        """
        self._session = session
        self._topic = topic
        self._sub_id = sub_id

    def undeclare(self) -> None:
        """
        Unsubscribe from the broker topic.
        """
        self._session._undeclare(self._sub_id)


class _Publisher:
    """
    Provides a publish interface to a broker topic, mimicking the Zenoh Publisher API.
    """

    def __init__(self, session: "CloudSimZenohSession", topic: str) -> None:
        """
        Initialize a publisher with the session and topic.

        Parameters
        ----------
        session : CloudSimZenohSession
            The Zenoh session this publisher belongs to.
        topic : str
            The topic this publisher is associated with.
        """
        self._session = session
        self._topic = topic

    def put(self, payload: Union[bytes, "ZBytes"]) -> None:  # type: ignore
        """
        Publish a message payload to the broker topic.

        Parameters
        ----------
        payload : bytes or ZBytes
            The raw bytes payload to publish to the broker.
        """
        # Convert ZBytes to bytes if needed
        if not isinstance(payload, bytes):
            payload = payload.to_bytes()  # type: ignore
        self._session._publish_binary(self._topic, payload)

    def undeclare(self) -> None:
        """
        Undeclare (cleanup) this publisher.

        Note: In CloudSimZenohSession, publishers don't require explicit cleanup
        since they don't maintain persistent state. This method is provided for
        API compatibility with zenoh.Publisher.
        """
        # No-op for cloud session publishers - they're stateless
        logging.debug(f"Publisher undeclare called for topic {self._topic} (no-op for cloud session)")


class CloudSimZenohSession:
    """CloudSimZenohSession provides a Zenoh-like API for subscribing and publishing to the cloud broker."""

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
    ) -> None:
        """
        Initialize the CloudSimZenohSession.

        Parameters
        ----------
        url : str
            The WebSocket URL for the cloud broker.
        token : Optional[str]
            Optional authentication token for the broker. If not provided, the session will attempt unauthenticated access.
        """
        self._client = CloudZenohClient(url, token)
        self._subscription_topics: dict[str, str] = {}

    def declare_subscriber(
        self,
        topic: str,
        handler: Optional[Callable[[Any], None]],
    ) -> Optional[_Subscriber]:
        """
        Subscribe a handler to messages on `topic` via the broker.

        Parameters
        ----------
        topic : str
            The topic to subscribe to, e.g. "odom" or "image".
        handler : Callable[[Any], None]
            A callback function that takes a sample object as input and processes incoming messages from the broker.

        Returns
        -------
        Optional[_Subscriber]
            A subscriber object, or None if the topic is not mapped.
        """

        def _on_cdr(cdr: bytes, _topic=topic) -> None:
            if handler is None:
                return
            try:
                handler(_Sample(_topic, cdr))
            except Exception:
                logging.exception(f"CloudSimZenohSession handler raised on {_topic}")

        sub_id = self._client.declare_subscriber(
            topic=topic,
            callback=_on_cdr,
            binary=True,
        )
        self._subscription_topics[sub_id] = topic

        return _Subscriber(self, topic, sub_id)

    def declare_publisher(self, topic: str) -> _Publisher:
        """
        Return a publisher object whose `.put(bytes)` forwards to the broker.

        Parameters
        ----------
        topic : str
            The OM1 topic to publish to, e.g. "cmd_vel" or "joint_states".

        Returns
        -------
        _Publisher
            A publisher object, or None if the topic is not mapped.
        """
        return _Publisher(self, topic)

    def put(self, topic: str, payload: Union[bytes, ZBytes]) -> None:
        """
        Publish a message directly to a topic (zenoh.Session API compatibility).

        This method provides compatibility with zenoh.Session.put(topic, payload).
        It resolves the topic and publishes the payload directly.

        Parameters
        ----------
        topic : str
            The OM1 topic to publish to, e.g. "cmd_vel" or "joint_states".
        payload : bytes or ZBytes
            The raw bytes payload to publish.
        """
        if not isinstance(payload, bytes):
            payload = payload.to_bytes()
        self._publish_binary(topic, payload)

    def close(self) -> None:
        """Close the underlying broker connection."""
        self._client.close()

    def _undeclare(self, sub_id: str) -> None:
        """
        Unsubscribe from a broker topic by subscription ID.

        Parameters
        ----------
        sub_id : str
            The unique subscription ID returned by the broker when declaring the subscriber.
        """
        self._subscription_topics.pop(sub_id, None)
        self._client.undeclare_subscriber(sub_id)

    def _publish_binary(self, topic: str, cdr: bytes) -> None:
        """
        Publish a binary message to the broker topic.

        Parameters
        ----------
        topic : str
            The zenoh topic to publish to.
        cdr : bytes
            The raw bytes payload to publish.
        """
        self._client.publish_binary(topic, cdr)
