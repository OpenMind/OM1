import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests

from providers.ub_tts_provider import UbTtsProvider


def test_initialization_sets_url():
    """Test that initialization correctly sets the TTS URL."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    assert provider.tts_url == "http://localhost:8080/tts"
    provider.stop()


def test_initialization_sets_headers():
    """Test that initialization sets correct headers."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    assert provider.headers == {"Content-Type": "application/json"}
    provider.stop()


def test_initialization_creates_executor():
    """Test that initialization creates a ThreadPoolExecutor."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    assert isinstance(provider.executor, ThreadPoolExecutor)
    provider.stop()


def test_start_method():
    """Test start method logs appropriately."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    with patch("providers.ub_tts_provider.logging.info") as mock_log:
        provider.start()
        mock_log.assert_called_with("Ubtech TTS Provider started.")

    provider.stop()


def test_stop_method():
    """Test stop method shuts down executor."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    executor_mock = MagicMock()
    provider.executor = executor_mock

    provider.stop()

    executor_mock.shutdown.assert_called_once_with(wait=True)


# Tests for async message processing


def test_adding_pending_message_submits_to_executor():
    """Test that adding_pending_message submits work to the executor."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    executor_mock = MagicMock()
    provider.executor = executor_mock

    provider.adding_pending_message("Hello world")

    executor_mock.submit.assert_called_once()
    args = executor_mock.submit.call_args[0]
    assert args[0] == provider._speak_workder
    assert args[1] == "Hello world"
    assert args[2] is True  # interrupt default
    assert args[3] == 0  # timestamp default

    provider.stop()


def test_adding_pending_message_with_custom_parameters():
    """Test that adding_pending_message correctly passes custom parameters."""
    provider = UbTtsProvider("http://localhost:8080/tts")
    executor_mock = MagicMock()
    provider.executor = executor_mock

    provider.adding_pending_message("Custom message", interrupt=False, timestamp=12345)

    executor_mock.submit.assert_called_once()
    args = executor_mock.submit.call_args[0]
    assert args[0] == provider._speak_workder
    assert args[1] == "Custom message"
    assert args[2] is False  # interrupt=False
    assert args[3] == 12345  # timestamp=12345

    provider.stop()


def test_adding_pending_message_integration():
    """Test integration of adding_pending_message with actual worker."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    with patch("providers.ub_tts_provider.requests.put", return_value=mock_response):
        provider.adding_pending_message("Hello world")

        time.sleep(0.1)

    provider.stop()


def test_speak_worker_success():
    """Test successful TTS speak request through worker."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    with patch(
        "providers.ub_tts_provider.requests.put", return_value=mock_response
    ) as mock_put:
        result = provider._speak_workder("Hello world")

        assert result is True
        mock_put.assert_called_once()
        call_kwargs = mock_put.call_args[1]
        assert call_kwargs["url"] == "http://localhost:8080/tts"
        assert call_kwargs["timeout"] == 5

    provider.stop()


def test_speak_worker_with_parameters():
    """Test speak worker with custom interrupt and timestamp parameters."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    with patch(
        "providers.ub_tts_provider.requests.put", return_value=mock_response
    ) as mock_put:
        result = provider._speak_workder("Hello", interrupt=False, timestamp=12345)

        assert result is True
        call_data = mock_put.call_args[1]["data"]
        assert '"interrupt": false' in call_data
        assert '"timestamp": 12345' in call_data

    provider.stop()


def test_speak_worker_default_parameters():
    """Test speak worker with default parameters."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_response.raise_for_status = MagicMock()

    with patch(
        "providers.ub_tts_provider.requests.put", return_value=mock_response
    ) as mock_put:
        result = provider._speak_workder("Hello")

        assert result is True
        call_data = mock_put.call_args[1]["data"]
        assert '"interrupt": true' in call_data
        assert '"timestamp": 0' in call_data

    provider.stop()


def test_speak_worker_failure_non_zero_code():
    """Test speak worker returns False when response code is non-zero."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 1, "error": "TTS busy"}
    mock_response.raise_for_status = MagicMock()

    with patch("providers.ub_tts_provider.requests.put", return_value=mock_response):
        result = provider._speak_workder("Hello")
        assert result is False

    provider.stop()


def test_speak_worker_request_exception():
    """Test speak worker handles request exceptions gracefully."""

    provider = UbTtsProvider("http://localhost:8080/tts")

    with (
        patch(
            "providers.ub_tts_provider.requests.put",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ),
        patch("providers.ub_tts_provider.logging.error") as mock_log,
    ):
        result = provider._speak_workder("Hello")
        assert result is False
        mock_log.assert_called_once()
        assert "Failed to send TTS command" in mock_log.call_args[0][0]

    provider.stop()


def test_speak_worker_timeout_exception():
    """Test speak worker handles timeout exceptions gracefully."""

    provider = UbTtsProvider("http://localhost:8080/tts")

    with (
        patch(
            "providers.ub_tts_provider.requests.put",
            side_effect=requests.exceptions.Timeout("Request timed out"),
        ),
        patch("providers.ub_tts_provider.logging.error") as mock_log,
    ):
        result = provider._speak_workder("Hello")
        assert result is False
        mock_log.assert_called_once()

    provider.stop()


def test_get_status_success():
    """Test successful status retrieval."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0, "status": "run"}

    with patch(
        "providers.ub_tts_provider.requests.get", return_value=mock_response
    ) as mock_get:
        result = provider.get_tts_status(12345)

        assert result == "run"
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"] == {"timestamp": 12345}
        assert call_kwargs["timeout"] == 2

    provider.stop()


def test_get_status_all_possible_values():
    """Test all possible status values."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    statuses = ["build", "wait", "run", "idle"]

    for status in statuses:
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "status": status}

        with patch(
            "providers.ub_tts_provider.requests.get", return_value=mock_response
        ):
            result = provider.get_tts_status(12345)
            assert result == status

    provider.stop()


def test_get_status_idle():
    """Test status returns idle when TTS is not active."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0, "status": "idle"}

    with patch("providers.ub_tts_provider.requests.get", return_value=mock_response):
        result = provider.get_tts_status(0)
        assert result == "idle"

    provider.stop()


def test_get_status_non_zero_code():
    """Test status returns error when response code is non-zero."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 1}

    with patch("providers.ub_tts_provider.requests.get", return_value=mock_response):
        result = provider.get_tts_status(12345)
        assert result == "error"

    provider.stop()


def test_get_status_request_exception():
    """Test status returns error on request exception."""

    provider = UbTtsProvider("http://localhost:8080/tts")

    with patch(
        "providers.ub_tts_provider.requests.get",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        result = provider.get_tts_status(12345)
        assert result == "error"

    provider.stop()


def test_get_status_missing_status_field():
    """Test status returns error when status field is missing."""
    provider = UbTtsProvider("http://localhost:8080/tts")

    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}  # No status field

    with patch("providers.ub_tts_provider.requests.get", return_value=mock_response):
        result = provider.get_tts_status(12345)
        assert result == "error"

    provider.stop()


def test_get_status_timeout_exception():
    """Test status returns error on timeout."""

    provider = UbTtsProvider("http://localhost:8080/tts")

    with patch(
        "providers.ub_tts_provider.requests.get",
        side_effect=requests.exceptions.Timeout("Request timed out"),
    ):
        result = provider.get_tts_status(12345)
        assert result == "error"

    provider.stop()
