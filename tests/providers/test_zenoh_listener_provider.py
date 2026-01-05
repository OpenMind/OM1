from unittest.mock import MagicMock, patch

import pytest

from providers.zenoh_listener_provider import ZenohListenerProvider


@pytest.fixture
def mock_session():
    session = MagicMock()
    subscriber = MagicMock()
    session.declare_subscriber.return_value = subscriber
    session.close = MagicMock()
    return session


@pytest.fixture
def listener(mock_session):
    with patch(
        "providers.zenoh_listener_provider.open_zenoh_session",
        return_value=mock_session,
    ):
        provider = ZenohListenerProvider(topic="test-topic")
        yield provider
        provider.stop()


def test_start_and_stop(listener, mock_session):
    listener.start()
    assert listener.running is True

    listener.stop()
    assert listener.running is False
    mock_session.close.assert_called_once()


def test_register_callback(listener, mock_session):
    callback = MagicMock()
    listener.start(callback)

    mock_session.declare_subscriber.assert_called_once()
    args, _ = mock_session.declare_subscriber.call_args
    assert args[0] == "test-topic"


def test_double_start_is_safe(listener):
    listener.start()
    listener.start()  # should not raise
    assert listener.running is True


def test_stop_without_start(listener):
    listener.stop()  # should not raise
    assert listener.running is False
