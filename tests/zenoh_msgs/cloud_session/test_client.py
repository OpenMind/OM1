from unittest.mock import MagicMock, patch

import pytest

from zenoh_msgs.cloud_session.client import CloudZenohClient


@pytest.fixture
def stop_thread():
    """Patch threading.Thread so the background runner doesn't actually run."""
    with patch("zenoh_msgs.cloud_session.client.threading.Thread") as mock_thread_cls:
        thread = MagicMock()
        mock_thread_cls.return_value = thread
        yield thread


def test_with_token_no_query_string():
    out = CloudZenohClient._with_token("wss://x.com/api", "abc")
    assert out == "wss://x.com/api?api_key=abc"


def test_with_token_existing_query_string():
    out = CloudZenohClient._with_token("wss://x.com/api?foo=bar", "abc")
    assert out == "wss://x.com/api?foo=bar&api_key=abc"


def test_init_no_token_keeps_url(stop_thread):
    c = CloudZenohClient("wss://x.com/api")
    assert c._url == "wss://x.com/api"
    stop_thread.start.assert_called_once()


def test_init_with_token_appends_query(stop_thread):
    c = CloudZenohClient("wss://x.com/api", token="zzz")
    assert "api_key=zzz" in c._url


def test_declare_subscriber_returns_id_and_sends(stop_thread):
    c = CloudZenohClient("wss://x.com")
    cb = MagicMock()
    sub_id = c.declare_subscriber("odom", cb, binary=True)
    assert isinstance(sub_id, str) and len(sub_id) == 12
    assert c._subscriptions[sub_id] is cb
    assert c._subscription_is_binary[sub_id] is True
    assert c._subscription_specs[sub_id]["topic"] == "odom"


def test_undeclare_subscriber_removes(stop_thread):
    c = CloudZenohClient("wss://x.com")
    sub_id = c.declare_subscriber("odom", MagicMock(), binary=False)
    c.undeclare_subscriber(sub_id)
    assert sub_id not in c._subscriptions
    assert sub_id not in c._subscription_is_binary
    assert sub_id not in c._subscription_specs


def test_publish_emits_message(stop_thread):
    c = CloudZenohClient("wss://x.com")
    with patch.object(c, "_send") as mock_send:
        c.publish("cmd_vel", {"linear": {}})
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert msg["type"] == "publish"
        assert msg["topic"] == "cmd_vel"


def test_publish_binary_frames_correctly(stop_thread):
    c = CloudZenohClient("wss://x.com")
    with patch.object(c, "_send_bytes") as mock_send:
        c.publish_binary("odom", b"\x01\x02\x03")
        frame = mock_send.call_args[0][0]
        # 0x20 + 2-byte topic length (BE) + topic + payload
        assert frame[0:1] == b"\x20"
        topic_len = int.from_bytes(frame[1:3], "big")
        assert topic_len == 4
        assert frame[3 : 3 + topic_len] == b"odom"
        assert frame[3 + topic_len :] == b"\x01\x02\x03"


def test_publish_binary_rejects_long_topic(stop_thread):
    c = CloudZenohClient("wss://x.com")
    with pytest.raises(ValueError, match="topic too long"):
        c.publish_binary("a" * 70000, b"")


def test_send_skips_when_not_connected(stop_thread):
    c = CloudZenohClient("wss://x.com")
    c._websocket = None
    # Should be a no-op, not raise
    c._send({"type": "ping"})


def test_send_bytes_raises_when_disconnected(stop_thread):
    c = CloudZenohClient("wss://x.com")
    c._connected = MagicMock()
    c._connected.wait.return_value = False
    with pytest.raises(RuntimeError, match="timeout waiting"):
        c._send_bytes(b"x")


def test_send_bytes_raises_when_websocket_none(stop_thread):
    c = CloudZenohClient("wss://x.com")
    c._connected = MagicMock()
    c._connected.wait.return_value = True
    c._websocket = None
    with pytest.raises(RuntimeError, match="not connected"):
        c._send_bytes(b"x")


def test_dispatch_json_sample_calls_callback(stop_thread):
    c = CloudZenohClient("wss://x.com")
    cb = MagicMock()
    sub_id = c.declare_subscriber("odom", cb, binary=False)
    c._dispatch_json({"type": "sample", "subId": sub_id, "payload": {"x": 1}})
    cb.assert_called_once_with({"x": 1})


def test_dispatch_json_sample_callback_exception_doesnt_propagate(stop_thread):
    c = CloudZenohClient("wss://x.com")
    cb = MagicMock(side_effect=RuntimeError("boom"))
    sub_id = c.declare_subscriber("odom", cb, binary=False)
    # Should not raise
    c._dispatch_json({"type": "sample", "subId": sub_id, "payload": {}})


def test_dispatch_json_sample_unknown_subid_no_op(stop_thread):
    c = CloudZenohClient("wss://x.com")
    # No callback registered for "missing" — silent no-op
    c._dispatch_json({"type": "sample", "subId": "missing", "payload": {}})


def test_dispatch_json_other_types(stop_thread):
    c = CloudZenohClient("wss://x.com")
    # These just log and don't crash
    c._dispatch_json({"type": "ack", "id": "1"})
    c._dispatch_json({"type": "error", "message": "foo"})
    c._dispatch_json({"type": "pong"})
    c._dispatch_json({"type": "wat"})


def test_dispatch_binary_dispatches_to_callback(stop_thread):
    c = CloudZenohClient("wss://x.com")
    cb = MagicMock()
    sub_id = c.declare_subscriber("odom", cb, binary=True)
    sub_b = sub_id.encode("utf-8")
    payload = b"\xab\xcd"
    frame = bytes([0x10]) + len(sub_b).to_bytes(2, "big") + sub_b + payload
    c._dispatch_binary(frame)
    cb.assert_called_once_with(payload)


def test_dispatch_binary_unknown_type_logs(stop_thread):
    c = CloudZenohClient("wss://x.com")
    # Frame with type byte != 0x10 is logged but not dispatched.
    c._dispatch_binary(bytes([0x99]) + b"junk")  # short, also unknown type
    c._dispatch_binary(b"")  # empty


def test_dispatch_binary_malformed_length(stop_thread):
    c = CloudZenohClient("wss://x.com")
    # Sub id length claims 100 bytes but the frame doesn't contain them.
    c._dispatch_binary(bytes([0x10]) + (100).to_bytes(2, "big") + b"short")


def test_dispatch_binary_callback_exception_caught(stop_thread):
    c = CloudZenohClient("wss://x.com")
    cb = MagicMock(side_effect=RuntimeError("boom"))
    sub_id = c.declare_subscriber("odom", cb, binary=True)
    sub_b = sub_id.encode("utf-8")
    frame = bytes([0x10]) + len(sub_b).to_bytes(2, "big") + sub_b + b"data"
    # Should not raise
    c._dispatch_binary(frame)


def test_dispatch_binary_unknown_subid(stop_thread):
    c = CloudZenohClient("wss://x.com")
    sub_b = b"deadbeefcafe"
    frame = bytes([0x10]) + len(sub_b).to_bytes(2, "big") + sub_b + b"data"
    # Subscription not registered -> early return, no crash.
    c._dispatch_binary(frame)


def test_close_signals_stop(stop_thread):
    c = CloudZenohClient("wss://x.com")
    # Don't actually have a websocket; close should still set _stopped and join.
    c._websocket = None
    c.close()
    assert c._stopped.is_set()
    stop_thread.join.assert_called_once_with(timeout=5)
