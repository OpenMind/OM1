import time
from queue import Queue
from unittest.mock import MagicMock, patch

from providers.zenoh_publisher_provider import ZenohPublisherProvider


def test_initialization_success():
    """Test successful ZenohPublisherProvider initialization."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider(topic="test/topic")

        assert provider.session == mock_session
        assert provider.pub_topic == "test/topic"
        assert provider.running is False
        assert isinstance(provider._pending_messages, Queue)
        mock_zenoh.assert_called_once()


def test_initialization_default_topic():
    """Test initialization with default topic."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()

        assert provider.pub_topic == "speech"


def test_initialization_failure():
    """Test handling of initialization failure."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_zenoh.side_effect = Exception("Connection failed")

        provider = ZenohPublisherProvider()

        assert provider.session is None
        assert provider.running is False


def test_add_pending_message():
    """Test adding a message to the pending queue."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()

        provider.add_pending_message("test message")

        assert not provider._pending_messages.empty()
        msg = provider._pending_messages.get()
        assert msg["message"] == "test message"
        assert "time_stamp" in msg
        assert isinstance(msg["time_stamp"], float)


def test_publish_message_success():
    """Test publishing a message successfully."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider(topic="test/topic")

        msg = {"time_stamp": time.time(), "message": "test"}
        provider._publish_message(msg)

        mock_session.put.assert_called_once()
        call_args = mock_session.put.call_args
        assert call_args[0][0] == "test/topic"


def test_publish_message_without_session():
    """Test publishing when session is None."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_zenoh.side_effect = Exception("Connection failed")

        provider = ZenohPublisherProvider()

        msg = {"time_stamp": time.time(), "message": "test"}
        # Should not raise an exception, just log and return
        provider._publish_message(msg)


def test_start():
    """Test starting the publisher provider."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.start()

        assert provider.running is True
        assert provider._thread is not None
        assert provider._thread.daemon is True


def test_start_already_running():
    """Test starting when already running."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.start()

        first_thread = provider._thread

        # Try to start again
        provider.start()

        # Thread should not change
        assert provider._thread is first_thread


def test_stop():
    """Test stopping the publisher provider."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.start()

        assert provider.running is True

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()


def test_run_loop_processes_messages():
    """Test that the run loop processes pending messages."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.add_pending_message("test message 1")
        provider.add_pending_message("test message 2")

        provider.start()

        # Give thread time to process
        time.sleep(0.1)
        provider.stop()

        # Should have published both messages
        assert mock_session.put.call_count >= 2


def test_message_format():
    """Test that messages are formatted correctly."""
    with patch("providers.zenoh_publisher_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohPublisherProvider()
        provider.add_pending_message("test message")

        msg = provider._pending_messages.get()

        assert "time_stamp" in msg
        assert "message" in msg
        assert msg["message"] == "test message"
        assert isinstance(msg["time_stamp"], float)
