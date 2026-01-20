from unittest.mock import MagicMock, patch

import pytest

from providers.config_provider import ConfigProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ConfigProvider.reset()  # type: ignore
    yield

    try:
        provider = ConfigProvider()
        provider.stop()
    except Exception:
        pass

    ConfigProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    with patch("providers.config_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_publisher = MagicMock()
        mock_subscriber = MagicMock()
        mock_session_instance.declare_publisher.return_value = mock_publisher
        mock_session_instance.declare_subscriber.return_value = mock_subscriber
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance, mock_publisher, mock_subscriber


def test_initialization(mock_zenoh):
    mock_session, mock_session_instance, mock_publisher, mock_subscriber = mock_zenoh
    provider = ConfigProvider()

    assert provider.running
    assert provider.session == mock_session_instance
    assert provider.config_response_publisher == mock_publisher
    assert provider.config_request_subscriber == mock_subscriber

    mock_session_instance.declare_publisher.assert_called_once_with(
        "om/config/response"
    )
    mock_session_instance.declare_subscriber.assert_called_once()


def test_singleton_pattern():
    provider1 = ConfigProvider()
    provider2 = ConfigProvider()
    assert provider1 is provider2


def test_get_runtime_config_path(mock_zenoh):
    provider = ConfigProvider()
    config_path = provider._get_runtime_config_path()

    assert config_path.endswith(".runtime.json5")
    assert "config/memory" in config_path


def test_initialization_failure():
    with patch("providers.config_provider.open_zenoh_session") as mock_session:
        mock_session.side_effect = Exception("Connection failed")
        provider = ConfigProvider()

        assert not provider.running
        assert provider.session is None


def test_stop(mock_zenoh):
    _, mock_session_instance, _, _ = mock_zenoh
    provider = ConfigProvider()

    provider.stop()

    assert not provider.running
    mock_session_instance.close.assert_called_once()


def test_handle_config_request(mock_zenoh):
    """Test that config request handler is registered correctly."""
    _, mock_session_instance, _, _ = mock_zenoh
    ConfigProvider()

    call_args = mock_session_instance.declare_subscriber.call_args
    assert call_args[0][0] == "om/config/request"
    assert callable(call_args[0][1])
