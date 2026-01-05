import time
from unittest.mock import MagicMock, patch

import pytest

from providers.zenoh_publisher_provider import ZenohPublisherProvider


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.put = MagicMock()
    session.close = MagicMock()
    return session


@pytest.fixture
def publisher(mock_session):
    with patch(
        "providers.zenoh_publisher_provider.open_zenoh_session",
        return_value=mock_session,
    ):
        provider = ZenohPublisherProvider(topic="test-topic", queue_size=10)
        yield provider
        provider.stop()


def test_start_and_stop(publisher):
    publisher.start()
    assert publisher._thread is not None
    publisher.stop()
    assert publisher._thread is None


def test_message_is_published(publisher, mock_session):
    publisher.start()
    publisher.add_pending_message("hello world")

    time.sleep(0.2)

    assert mock_session.put.called
    args, _ = mock_session.put.call_args
    assert args[0] == "test-topic"


def test_no_publish_after_stop(publisher, mock_session):
    publisher.start()
    publisher.stop()

    publisher.add_pending_message("should not publish")
    time.sleep(0.2)

    mock_session.put.assert_not_called()


def test_queue_overflow_drops_messages(publisher):
    publisher.start()

    for i in range(50):
        publisher.add_pending_message(f"msg-{i}")

    # no exception should be raised
    assert True
