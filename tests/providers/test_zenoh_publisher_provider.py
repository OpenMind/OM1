from unittest.mock import MagicMock, patch
import time

import pytest

from providers.zenoh_publisher_provider import ZenohPublisherProvider


@pytest.fixture
def mock_zenoh_session():
    session = MagicMock()
    return session


@pytest.fixture
def provider(mock_zenoh_session):
    with patch(
        "providers.zenoh_publisher_provider.open_zenoh_session",
        return_value=mock_zenoh_session,
    ):
        provider = ZenohPublisherProvider(topic="test-topic")
        yield provider
        provider.stop()


def test_initialization_opens_session(mock_zenoh_session):
    with patch(
        "providers.zenoh_publisher_provider.open_zenoh_session",
        return_value=mock_zenoh_session,
    ):
        provider = ZenohPublisherProvider()
        assert provider.session is mock_zenoh_session
        assert provider.running is False


def test_add_pending_message(provider):
    result = provider.add_pending_message("hello world")
    assert result is True
    assert not provider._pending_messages.empty()


def test_add_pending_message_queue_full(mock_zenoh_session):
    with patch(
        "providers.zenoh_publisher_provider.open_zenoh_session",
        return_value=mock_zenoh_session,
    ):
        provider = ZenohPublisherProvider(max_queue_size=1)
        assert provider.add_pending_message("first") is True
        assert provider.add_pending_message("second") is False


def test_start_and_stop(provider):
    provider.start()
    assert provider.running is True
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()
    assert provider.running is False


def test_publish_message_called(provider, mock_zenoh_session):
    provider.start()
    provider.add_pending_message("test message")

    time.sleep(0.2)

    mock_zenoh_session.put.assert_called_once()
    provider.stop()


def test_no_publish_when_session_none(mock_zenoh_session):
    with patch(
        "providers.zenoh_publisher_provider.open_zenoh_session",
        side_effect=Exception("connection failed"),
    ):
        provider = ZenohPublisherProvider()
        provider.start()
        provider.add_pending_message("test")

        time.sleep(0.2)

        assert provider.session is None
        provider.stop()
