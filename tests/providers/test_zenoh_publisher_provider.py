import sys
import time
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_zenoh_mock(monkeypatch):
    monkeypatch.setitem(sys.modules, "zenoh", Mock())
    zenoh_mock = sys.modules["zenoh"]
    zenoh_mock.ZBytes = Mock()

    monkeypatch.setitem(sys.modules, "zenoh_msgs", Mock())
    zenoh_msgs_mock = sys.modules["zenoh_msgs"]
    zenoh_msgs_mock.open_zenoh_session = Mock()


class TestZenohPublisherProvider:

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_init_success(self, mock_open_session, mock_thread_class, monkeypatch):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        topic = "test_topic"
        provider = ZenohPublisherProvider(topic=topic)

        mock_open_session.assert_called_once()
        assert provider.session is mock_session
        assert provider.pub_topic == topic
        assert provider.running is False
        assert hasattr(provider, "_pending_messages")
        assert hasattr(provider, "_lock")
        assert provider._thread is None

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_init_failure(self, mock_open_session, mock_thread_class, monkeypatch):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        error_msg = "Connection failed"
        mock_open_session.side_effect = Exception(error_msg)

        topic = "test_topic"
        provider = ZenohPublisherProvider(topic=topic)

        mock_open_session.assert_called_once()
        assert provider.session is None
        assert provider.pub_topic == topic
        assert provider.running is False
        assert hasattr(provider, "_pending_messages")
        assert hasattr(provider, "_lock")
        assert provider._thread is None

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_add_pending_message(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohPublisherProvider()

        test_text = "Hello, Zenoh!"
        provider.add_pending_message(test_text)

        assert not provider._pending_messages.empty()
        queued_item = provider._pending_messages.get_nowait()
        assert isinstance(queued_item, dict)
        assert "time_stamp" in queued_item
        assert queued_item["message"] == test_text
        assert abs(queued_item["time_stamp"] - time.time()) < 1.0

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_start(self, mock_open_session, mock_thread_class, monkeypatch):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        provider = ZenohPublisherProvider()

        assert not provider.running
        assert provider._thread is None

        provider.start()

        assert provider.running
        assert provider._thread is mock_thread_instance
        mock_thread_instance.start.assert_called_once()

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_start_already_running(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        provider = ZenohPublisherProvider()
        provider.running = True
        original_thread = MagicMock()
        provider._thread = original_thread

        provider.start()

        assert provider.running
        assert provider._thread is original_thread
        mock_thread_instance.start.assert_not_called()

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_publish_message_with_session(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.session = mock_session

        test_msg = {"time_stamp": time.time(), "message": "test"}

        provider._publish_message(test_msg)

        mock_session.put.assert_called_once_with(provider.pub_topic, ANY)

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_publish_message_without_session(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_open_session.side_effect = Exception("Init Failed")
        provider = ZenohPublisherProvider()
        # Ensure session is None
        assert provider.session is None

        test_msg = {"time_stamp": time.time(), "message": "test"}

        # Should not raise an exception
        provider._publish_message(test_msg)

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_stop_when_running_with_session(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        provider = ZenohPublisherProvider()
        provider.session = mock_session
        provider._thread = mock_thread_instance
        provider.running = True

        provider.stop()

        assert not provider.running
        mock_thread_instance.join.assert_called_once_with(timeout=5)
        mock_session.close.assert_called_once()

    @patch("src.providers.zenoh_publisher_provider.threading.Thread")
    @patch("src.providers.zenoh_publisher_provider.open_zenoh_session")
    def test_stop_when_not_running_no_session(
        self, mock_open_session, mock_thread_class, monkeypatch
    ):
        from src.providers.zenoh_publisher_provider import ZenohPublisherProvider

        mock_open_session.side_effect = Exception("No session")
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        provider = ZenohPublisherProvider()
        provider._thread = mock_thread_instance
        provider.running = False

        provider.stop()

        assert not provider.running
        mock_thread_instance.join.assert_called_once_with(timeout=5)
        # Session is None, so session.close() should not be called.
        # Assert based on the implementation detail that close isn't called on a None session.
        # We confirmed session is None earlier.
        assert provider.session is None
