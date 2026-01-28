import json
import os
from unittest.mock import MagicMock, mock_open, patch

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
    """Test that ConfigProvider initializes correctly with Zenoh session."""
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


def test_singleton_pattern(mock_zenoh):
    """Test that ConfigProvider follows singleton pattern."""
    provider1 = ConfigProvider()
    provider2 = ConfigProvider()
    assert provider1 is provider2


def test_get_runtime_config_path(mock_zenoh):
    """Test that config path is correctly resolved."""
    provider = ConfigProvider()
    config_path = provider._get_runtime_config_path()

    assert config_path.endswith(".runtime.json5")
    assert "config/memory" in config_path


def test_initialization_failure():
    """Test that initialization handles Zenoh session failures gracefully."""
    with patch("providers.config_provider.open_zenoh_session") as mock_session:
        mock_session.side_effect = Exception("Connection failed")
        provider = ConfigProvider()

        assert not provider.running
        assert provider.session is None
        assert provider.config_response_publisher is None
        assert provider.config_request_subscriber is None


def test_stop_cleanup_resources(mock_zenoh):
    """Test that stop() correctly cleans up all resources."""
    _, mock_session_instance, mock_publisher, mock_subscriber = mock_zenoh
    provider = ConfigProvider()

    provider.stop()

    assert not provider.running
    assert provider.config_request_subscriber is None
    assert provider.config_response_publisher is None
    mock_subscriber.undeclare.assert_called_once()
    mock_publisher.undeclare.assert_called_once()
    mock_session_instance.close.assert_called_once()


def test_stop_when_not_running(mock_zenoh):
    """Test that stop() handles case when provider is not running."""
    provider = ConfigProvider()
    provider.running = False

    provider.stop()

    assert not provider.running


def test_get_config_snapshot_file_exists(mock_zenoh):
    """Test that _get_config_snapshot reads config file when it exists."""
    provider = ConfigProvider()
    test_config = {"key": "value", "nested": {"data": 123}}

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=json.dumps(test_config))):
            with patch("providers.config_provider.json5.load", return_value=test_config):
                result = provider._get_config_snapshot()

    assert result == test_config


def test_get_config_snapshot_file_not_exists(mock_zenoh):
    """Test that _get_config_snapshot returns empty dict when file doesn't exist."""
    provider = ConfigProvider()

    with patch("os.path.exists", return_value=False):
        result = provider._get_config_snapshot()

    assert result == {}


def test_get_config_snapshot_read_error(mock_zenoh):
    """Test that _get_config_snapshot handles file read errors gracefully."""
    provider = ConfigProvider()

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = provider._get_config_snapshot()

    assert result == {}


def test_handle_config_request_get_config(mock_zenoh):
    """Test handling of get_config request."""
    _, _, mock_publisher, _ = mock_zenoh
    provider = ConfigProvider()

    mock_sample = MagicMock()
    mock_request = MagicMock()
    mock_request.request_id = "test-request-id"
    mock_request.config = None

    with patch("providers.config_provider.ConfigRequest.deserialize", return_value=mock_request):
        with patch.object(provider, "_send_config_response") as mock_send:
            provider._handle_config_request(mock_sample)

    mock_send.assert_called_once_with("test-request-id")


def test_handle_config_request_set_config(mock_zenoh):
    """Test handling of set_config request."""
    _, _, _, _ = mock_zenoh
    provider = ConfigProvider()

    mock_sample = MagicMock()
    mock_request = MagicMock()
    mock_request.request_id = "test-request-id"
    mock_config = MagicMock()
    mock_config.data = '{"new_key": "new_value"}'
    mock_request.config = mock_config

    with patch("providers.config_provider.ConfigRequest.deserialize", return_value=mock_request):
        with patch.object(provider, "_handle_set_config") as mock_set:
            provider._handle_config_request(mock_sample)

    mock_set.assert_called_once_with("test-request-id", '{"new_key": "new_value"}')


def test_handle_config_request_deserialize_error(mock_zenoh):
    """Test that _handle_config_request handles deserialization errors."""
    provider = ConfigProvider()

    mock_sample = MagicMock()

    with patch("providers.config_provider.ConfigRequest.deserialize", side_effect=Exception("Invalid data")):
        provider._handle_config_request(mock_sample)

    # Should not raise, error should be logged


def test_handle_set_config_success(mock_zenoh):
    """Test successful config update."""
    _, _, mock_publisher, _ = mock_zenoh
    provider = ConfigProvider()
    provider.config_path = "/tmp/test.runtime.json5"

    new_config = {"updated": True}
    config_str = json.dumps(new_config)

    with patch("providers.config_provider.json5.loads", return_value=new_config):
        with patch("builtins.open", mock_open()):
            with patch("os.rename") as mock_rename:
                with patch.object(provider, "_send_config_response") as mock_send:
                    provider._handle_set_config("test-request-id", config_str)

    mock_rename.assert_called_once()
    mock_send.assert_called_once_with("test-request-id")


def test_handle_set_config_json_error(mock_zenoh):
    """Test that _handle_set_config handles JSON parsing errors."""
    _, _, mock_publisher, _ = mock_zenoh
    provider = ConfigProvider()

    with patch("providers.config_provider.json5.loads", side_effect=ValueError("Invalid JSON")):
        with patch.object(provider, "_send_error_response") as mock_error:
            provider._handle_set_config("test-request-id", "invalid json")

    mock_error.assert_called_once()


def test_send_config_response_success(mock_zenoh):
    """Test successful config response sending."""
    _, _, mock_publisher, _ = mock_zenoh
    provider = ConfigProvider()

    test_config = {"test": "data"}
    request_id = "test-request-id"

    with patch.object(provider, "_get_config_snapshot", return_value=test_config):
        with patch("providers.config_provider.prepare_header") as mock_header:
            with patch("providers.config_provider.ConfigResponse") as mock_response_class:
                mock_response = MagicMock()
                mock_response.serialize.return_value = b"serialized"
                mock_response_class.return_value = mock_response

                provider._send_config_response(request_id)

    mock_publisher.put.assert_called_once()


def test_send_config_response_no_publisher(mock_zenoh):
    """Test that _send_config_response handles missing publisher."""
    provider = ConfigProvider()
    provider.config_response_publisher = None

    with patch.object(provider, "_get_config_snapshot", return_value={}):
        provider._send_config_response("test-request-id")

    # Should not raise, error should be logged


def test_send_error_response_success(mock_zenoh):
    """Test successful error response sending."""
    _, _, mock_publisher, _ = mock_zenoh
    provider = ConfigProvider()

    with patch("providers.config_provider.prepare_header") as mock_header:
        with patch("providers.config_provider.ConfigResponse") as mock_response_class:
            mock_response = MagicMock()
            mock_response.serialize.return_value = b"serialized"
            mock_response_class.return_value = mock_response

            provider._send_error_response("test-request-id", "Error message")

    mock_publisher.put.assert_called_once()


def test_send_error_response_no_publisher(mock_zenoh):
    """Test that _send_error_response handles missing publisher."""
    provider = ConfigProvider()
    provider.config_response_publisher = None

    provider._send_error_response("test-request-id", "Error message")

    # Should not raise, error should be logged
