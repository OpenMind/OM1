from unittest.mock import MagicMock, patch

from providers.zenoh_listener_provider import ZenohListenerProvider


def test_initialization_success():
    """Test successful ZenohListenerProvider initialization."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider(topic="test/topic")

        assert provider.session == mock_session
        assert provider.sub_topic == "test/topic"
        assert provider.running is False
        mock_zenoh.assert_called_once()


def test_initialization_default_topic():
    """Test initialization with default topic."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider()

        assert provider.sub_topic == "speech"


def test_initialization_failure():
    """Test handling of initialization failure."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_zenoh.side_effect = Exception("Connection failed")

        provider = ZenohListenerProvider()

        assert provider.session is None
        assert provider.running is False


def test_register_message_callback_success():
    """Test registering a message callback."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_subscriber = MagicMock()
        mock_session.declare_subscriber.return_value = mock_subscriber
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider(topic="test/topic")

        def callback(sample):
            pass

        provider.register_message_callback(callback)

        mock_session.declare_subscriber.assert_called_once_with("test/topic", callback)


def test_register_message_callback_without_session():
    """Test registering callback when session is None."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_zenoh.side_effect = Exception("Connection failed")

        provider = ZenohListenerProvider()

        def callback(sample):
            pass

        # Should not raise an exception, just log an error
        provider.register_message_callback(callback)


def test_start_without_callback():
    """Test starting provider without callback."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.start()

        assert provider.running is True


def test_start_with_callback():
    """Test starting provider with callback."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_subscriber = MagicMock()
        mock_session.declare_subscriber.return_value = mock_subscriber
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider(topic="test/topic")

        def callback(sample):
            pass

        provider.start(message_callback=callback)

        assert provider.running is True
        mock_session.declare_subscriber.assert_called_once_with("test/topic", callback)


def test_start_already_running():
    """Test starting provider when already running."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.start()

        provider.start()

        assert provider.running is True


def test_stop():
    """Test stopping the provider."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.start()

        assert provider.running is True

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()


def test_stop_without_session():
    """Test stopping when session is None."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh:
        mock_zenoh.side_effect = Exception("Connection failed")

        provider = ZenohListenerProvider()
        provider.running = True

        provider.stop()

        assert provider.running is False
