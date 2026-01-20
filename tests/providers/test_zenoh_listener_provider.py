import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_zenoh_mock(monkeypatch):
    monkeypatch.setitem(sys.modules, "zenoh", Mock())

    monkeypatch.setitem(sys.modules, "zenoh_msgs", Mock())
    zenoh_msgs_mock = sys.modules["zenoh_msgs"]
    zenoh_msgs_mock.open_zenoh_session = Mock()


class TestZenohListenerProvider:

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_init_success(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        topic = "test_topic"
        provider = ZenohListenerProvider(topic=topic)

        mock_open_session.assert_called_once()
        assert provider.session is mock_session
        assert provider.sub_topic == topic
        assert provider.running is False

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_init_failure(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        error_msg = "Connection failed"
        mock_open_session.side_effect = Exception(error_msg)

        topic = "test_topic"
        provider = ZenohListenerProvider(topic=topic)

        mock_open_session.assert_called_once()
        assert provider.session is None
        assert provider.sub_topic == topic
        assert provider.running is False

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_success(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.session = mock_session
        callback = Mock()

        provider.register_message_callback(callback)

        mock_session.declare_subscriber.assert_called_once_with("speech", callback)

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_no_session(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_open_session.side_effect = Exception("Init Failed")
        provider = ZenohListenerProvider()
        assert provider.session is None
        callback = Mock()

        provider.register_message_callback(callback)

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_none(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.session = mock_session

        provider.register_message_callback(None)

        mock_session.declare_subscriber.assert_called_once_with("speech", None)

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_without_callback(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        provider.start()

        assert provider.running is True

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_with_callback(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        callback = Mock()

        provider.start(message_callback=callback)

        assert provider.running is True
        mock_session.declare_subscriber.assert_called_once_with("speech", callback)

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_already_running(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.running = True
        original_running_state = provider.running

        provider.start()

        assert provider.running is original_running_state

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_with_none_callback(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        provider.start(message_callback=None)

        assert provider.running is True
        # declare_subscriber should not be called if callback is None
        mock_session.declare_subscriber.assert_not_called()

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_with_session(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.session = mock_session
        provider.running = True

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_without_session(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_open_session.side_effect = Exception("Connection failed")
        provider = ZenohListenerProvider()
        assert provider.session is None
        provider.running = True

        provider.stop()

        assert provider.running is False

    @patch("src.providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_already_stopped(self, mock_open_session, monkeypatch):
        from src.providers.zenoh_listener_provider import ZenohListenerProvider

        mock_session = MagicMock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.running = False

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()
