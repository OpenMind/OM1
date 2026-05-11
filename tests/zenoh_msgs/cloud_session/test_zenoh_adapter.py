from unittest.mock import MagicMock, patch

import pytest

from zenoh_msgs.cloud_session.zenoh_adapter import (
    CloudSimZenohSession,
    _Payload,
    _Publisher,
    _Sample,
    _Subscriber,
)


def test_payload_to_bytes():
    p = _Payload(b"\x01\x02")
    assert p.to_bytes() == b"\x01\x02"


def test_sample_construction():
    s = _Sample("odom", b"\xde\xad")
    assert s.topic == "odom"
    assert s.payload.to_bytes() == b"\xde\xad"


def test_subscriber_undeclare_calls_session():
    session = MagicMock()
    sub = _Subscriber(session, "odom", "id123")
    sub.undeclare()
    session._undeclare.assert_called_once_with("id123")


def test_publisher_put_with_bytes():
    session = MagicMock()
    pub = _Publisher(session, "cmd_vel")
    pub.put(b"\x01\x02")
    session._publish_binary.assert_called_once_with("cmd_vel", b"\x01\x02")


def test_publisher_put_converts_zbytes_like():
    session = MagicMock()
    pub = _Publisher(session, "cmd_vel")
    zbytes = MagicMock()
    zbytes.to_bytes.return_value = b"\xaa\xbb"
    pub.put(zbytes)
    session._publish_binary.assert_called_once_with("cmd_vel", b"\xaa\xbb")


def test_publisher_undeclare_is_noop():
    session = MagicMock()
    pub = _Publisher(session, "cmd_vel")
    # No exception, no call to session.
    pub.undeclare()


@pytest.fixture
def session():
    """Return a CloudSimZenohSession with the websocket client mocked out."""
    with patch("zenoh_msgs.cloud_session.zenoh_adapter.CloudZenohClient") as mock_client_class:
        client = MagicMock()
        mock_client_class.return_value = client
        s = CloudSimZenohSession("wss://x.com", token="tok")
        # expose for assertions
        s._mock_client = client  # type: ignore[attr-defined]
        yield s


def test_declare_subscriber_returns_subscriber(session):
    session._mock_client.declare_subscriber.return_value = "subID"
    sub = session.declare_subscriber("odom", lambda _s: None)
    assert isinstance(sub, _Subscriber)
    assert session._subscription_topics["subID"] == "odom"


def test_subscriber_callback_wraps_in_sample(session):
    captured = []

    def handler(sample):
        captured.append(sample)

    session._mock_client.declare_subscriber.return_value = "subID"
    session.declare_subscriber("odom", handler)
    # Recover the inner callback the client was called with.
    inner_cb = session._mock_client.declare_subscriber.call_args.kwargs["callback"]
    inner_cb(b"\x42")
    assert len(captured) == 1
    assert captured[0].topic == "odom"
    assert captured[0].payload.to_bytes() == b"\x42"


def test_subscriber_callback_swallows_handler_exceptions(session):
    def handler(_sample):
        raise RuntimeError("boom")

    session._mock_client.declare_subscriber.return_value = "subID"
    session.declare_subscriber("odom", handler)
    inner_cb = session._mock_client.declare_subscriber.call_args.kwargs["callback"]
    # Should not raise
    inner_cb(b"\x00")


def test_subscriber_callback_none_handler_noop(session):
    session._mock_client.declare_subscriber.return_value = "subID"
    session.declare_subscriber("odom", None)
    inner_cb = session._mock_client.declare_subscriber.call_args.kwargs["callback"]
    # Should not raise even though handler is None
    inner_cb(b"\x00")


def test_declare_publisher_returns_publisher(session):
    pub = session.declare_publisher("cmd_vel")
    assert isinstance(pub, _Publisher)


def test_put_publishes_bytes(session):
    session.put("cmd_vel", b"\x01\x02")
    session._mock_client.publish_binary.assert_called_once_with("cmd_vel", b"\x01\x02")


def test_put_converts_zbytes_like(session):
    payload = MagicMock()
    payload.to_bytes.return_value = b"\xff"
    session.put("cmd_vel", payload)
    session._mock_client.publish_binary.assert_called_once_with("cmd_vel", b"\xff")


def test_close_closes_client(session):
    session.close()
    session._mock_client.close.assert_called_once()


def test_undeclare_clears_state_and_calls_client(session):
    session._mock_client.declare_subscriber.return_value = "subID"
    session.declare_subscriber("odom", lambda _s: None)
    session._undeclare("subID")
    assert "subID" not in session._subscription_topics
    session._mock_client.undeclare_subscriber.assert_called_once_with("subID")


def test_publish_binary_helper(session):
    session._publish_binary("cmd_vel", b"\xaa")
    session._mock_client.publish_binary.assert_called_once_with("cmd_vel", b"\xaa")
