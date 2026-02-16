import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from providers.ubtech_asr_provider import UbtechASRProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UbtechASRProvider.reset()  # type: ignore
    yield

    try:
        # Get instance by accessing the singleton class's _singleton_instance
        provider = UbtechASRProvider._singleton_class._singleton_instance  # type: ignore
        if provider:
            provider.stop()
    except Exception:
        pass

    UbtechASRProvider.reset()  # type: ignore


@pytest.fixture
def mock_requests():
    """Mock requests.Session for UbtechASRProvider."""
    with patch("providers.ubtech_asr_provider.requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        yield mock_session


def test_initialization(mock_requests):
    """Test UbtechASRProvider initialization."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100", language_code="en")

    assert provider.robot_ip == "192.168.1.100"
    assert provider.language == "en"
    assert provider.basic_url == "http://192.168.1.100:9090/v1/"
    assert provider.running is False
    assert provider.paused is False
    assert provider.just_resumed is False
    assert UbtechASRProvider() == provider  # type: ignore


def test_singleton_pattern(mock_requests):
    """Test that UbtechASRProvider follows singleton pattern."""
    provider1 = UbtechASRProvider(robot_ip="192.168.1.100")
    provider2 = UbtechASRProvider(robot_ip="192.168.1.101")
    assert provider1 is provider2


def test_register_message_callback(mock_requests):
    """Test registering message callback."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    def callback(text):
        pass

    provider.register_message_callback(callback)

    assert provider._message_callback == callback


def test_start(mock_requests):
    """Test starting the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    provider.start()

    assert provider.running is True
    assert provider._thread is not None
    assert provider._thread.daemon is True


def test_start_already_running(mock_requests):
    """Test starting when already running."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    provider.start()
    first_thread = provider._thread

    provider.start()

    assert provider._thread is first_thread


def test_stop(mock_requests):
    """Test stopping the ASR provider."""
    with patch.object(UbtechASRProvider._singleton_class, "_stop_voice_iat") as mock_stop_iat:  # type: ignore
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.start()

        assert provider.running is True

        provider.stop()

        assert provider.running is False
        assert mock_stop_iat.called


def test_stop_when_not_running(mock_requests):
    """Test stopping when not running."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")
    provider.stop()

    assert provider.running is False


def test_pause(mock_requests):
    """Test pausing the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    assert provider.paused is False

    provider.pause()

    assert provider.paused is True


def test_resume(mock_requests):
    """Test resuming the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")
    provider.pause()

    assert provider.paused is True
    assert provider.just_resumed is False

    provider.resume()

    assert provider.paused is False
    assert provider.just_resumed is True


def test_pause_resume_cycle(mock_requests):
    """Test pause and resume cycle."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    assert provider.paused is False
    assert provider.just_resumed is False

    provider.pause()
    assert provider.paused is True

    provider.resume()
    assert provider.paused is False
    assert provider.just_resumed is True


def test_language_setting(mock_requests):
    """Test language code setting."""
    provider_en = UbtechASRProvider(robot_ip="192.168.1.100", language_code="en")
    assert provider_en.language == "en"

    UbtechASRProvider.reset()  # type: ignore

    provider_zh = UbtechASRProvider(robot_ip="192.168.1.100", language_code="zh")
    assert provider_zh.language == "zh"


def test_session_headers(mock_requests):
    """Test that session headers are set correctly."""
    UbtechASRProvider(robot_ip="192.168.1.100")

    expected_headers = {"Content-Type": "application/json"}
    mock_requests.headers.update.assert_called_once_with(expected_headers)


class TestUbtechASRProviderInternal:
    """Test internal methods and edge cases."""

    def test_run_just_resumed(self, mock_requests):
        """Test _run when just_resumed is True."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.just_resumed = True
        provider.paused = False
        provider.running = True

        with (
            patch.object(
                provider, "_get_single_utterance", return_value=None
            ) as mock_get,
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "pause") as mock_pause,
        ):

            def stop_after_first(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_first
            provider._run()

            mock_sleep.assert_any_call(1.0)
            assert provider.just_resumed is False
            mock_get.assert_called_once()
            mock_pause.assert_called_once()

    def test_run_paused(self, mock_requests):
        """Test _run when paused is True."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.paused = True
        provider.running = True

        with (
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "_get_single_utterance") as mock_get,
        ):

            def stop_after_pause(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_pause
            provider._run()
            mock_sleep.assert_called_with(0.1)
            mock_get.assert_not_called()

    def test_run_with_text(self, mock_requests):
        """Test _run when a text is successfully obtained."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.running = True
        callback = MagicMock()
        provider.register_message_callback(callback)

        with (
            patch.object(
                provider, "_get_single_utterance", return_value="hello"
            ) as mock_get,
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "pause") as mock_pause,
        ):

            def stop_after_first(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_first
            provider._run()

            mock_get.assert_called_once()
            callback.assert_called_once_with("hello")
            mock_pause.assert_called_once()
            mock_sleep.assert_any_call(2.0)

    def test_run_with_no_text(self, mock_requests):
        """Test _run when no text is obtained."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.running = True

        with (
            patch.object(
                provider, "_get_single_utterance", return_value=None
            ) as mock_get,
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "pause") as mock_pause,
        ):

            def stop_after_first(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_first
            provider._run()

            mock_get.assert_called_once()
            mock_pause.assert_called_once()
            mock_sleep.assert_any_call(0.5)

    def test_run_with_request_exception(self, mock_requests):
        """Test _run when a RequestException occurs."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.running = True

        with (
            patch.object(
                provider,
                "_get_single_utterance",
                side_effect=requests.RequestException("error"),
            ) as mock_get,
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "pause") as mock_pause,
            patch("logging.error") as mock_log,
        ):

            def stop_after_first(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_first
            provider._run()

            mock_get.assert_called_once()
            mock_pause.assert_called_once()
            mock_log.assert_called_with(
                "UbtechASRProvider: RequestException during _get_single_utterance: error"
            )
            mock_sleep.assert_any_call(0.5)

    def test_run_with_general_exception(self, mock_requests):
        """Test _run when a general exception occurs."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.running = True

        with (
            patch.object(
                provider, "_get_single_utterance", side_effect=Exception("unexpected")
            ) as mock_get,
            patch("time.sleep") as mock_sleep,
            patch.object(provider, "pause") as mock_pause,
            patch("logging.error") as mock_log,
        ):

            def stop_after_first(*args):
                provider.running = False

            mock_sleep.side_effect = stop_after_first
            provider._run()

            mock_get.assert_called_once()
            mock_pause.assert_called_once()
            mock_log.assert_called_with(
                "UbtechASRProvider: Unexpected error in _run's try block: unexpected",
                exc_info=True,
            )
            mock_sleep.assert_any_call(0.5)

    def test_get_single_utterance_success(self, mock_requests):
        """Test successful retrieval of a single utterance."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.running = True
        ts = 12345

        with (
            patch.object(provider, "_stop_voice_iat") as mock_stop,
            patch.object(provider, "_start_voice_iat", return_value=True) as mock_start,
            patch.object(provider, "_get_voice_iat") as mock_get,
            patch("time.sleep"),
            patch("time.time", return_value=ts),
        ):

            # Simulate two polling responses: first running, then idle with text
            mock_get.side_effect = [
                {"status": "running", "timestamp": ts},
                {
                    "status": "idle",
                    "timestamp": ts,
                    "code": 0,
                    "data": {"text": {"ws": [{"cw": [{"w": "hello"}]}]}},
                },
            ]

            result = provider._get_single_utterance()

            assert mock_start.called, "_start_voice_iat was not called"
            assert (
                mock_get.call_count == 2
            ), f"Expected 2 calls to _get_voice_iat, got {mock_get.call_count}"
            assert result == "hello"
            assert mock_stop.call_count == 2
            mock_start.assert_called_once_with(ts)

    def test_get_single_utterance_start_fails(self, mock_requests):
        """Test when _start_voice_iat fails."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        with (
            patch.object(provider, "_stop_voice_iat") as mock_stop,
            patch.object(
                provider, "_start_voice_iat", return_value=False
            ) as mock_start,
            patch("time.sleep"),
        ):
            result = provider._get_single_utterance()
            assert result is None
            # _stop_voice_iat is called twice: at the beginning and in finally
            assert mock_stop.call_count == 2
            mock_start.assert_called_once()

    def test_get_single_utterance_timeout(self, mock_requests):
        """Test when polling times out without getting idle status."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        ts = 12345
        with (
            patch.object(provider, "_stop_voice_iat") as mock_stop,
            patch.object(provider, "_start_voice_iat", return_value=True),
            patch.object(
                provider,
                "_get_voice_iat",
                return_value={"status": "running", "timestamp": ts},
            ) as mock_get,
            patch("time.sleep"),
            patch("time.time", return_value=ts),
        ):
            provider.running = True
            result = provider._get_single_utterance()
            assert result is None
            assert mock_get.call_count == 100
            assert mock_stop.call_count == 2

    def test_stop_voice_iat_http_error_500(self, mock_requests):
        """Test _stop_voice_iat retries on HTTP 500 errors."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        error_response = MagicMock()
        http_error_500 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=500)
        )
        error_response.raise_for_status.side_effect = http_error_500
        success_response = MagicMock()
        success_response.raise_for_status.return_value = None

        mock_delete = MagicMock(
            side_effect=[error_response, error_response, success_response]
        )
        with (
            patch.object(provider.session, "delete", mock_delete),
            patch("time.sleep") as mock_sleep,
        ):
            provider._stop_voice_iat()
            assert mock_delete.call_count == 3
            assert mock_sleep.call_count == 2

    def test_stop_voice_iat_http_error_400(self, mock_requests):
        """Test _stop_voice_iat handles non-retryable HTTP error."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        error_response = MagicMock()
        http_error_400 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=400)
        )
        error_response.raise_for_status.side_effect = http_error_400
        mock_delete = MagicMock(return_value=error_response)
        with (
            patch.object(provider.session, "delete", mock_delete),
            patch("logging.error") as mock_log,
        ):
            provider._stop_voice_iat()
            mock_delete.assert_called_once()
            mock_log.assert_called_once()
            args, _ = mock_log.call_args
            assert (
                "UbtechASRProvider: _stop_voice_iat failed with HTTPError (attempt 1): "
                in args[0]
            )

    def test_stop_voice_iat_request_exception(self, mock_requests):
        """Test _stop_voice_iat handles RequestException with retries."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        req_error = requests.RequestException("connection error")
        mock_delete = MagicMock(side_effect=req_error)
        with (
            patch.object(provider.session, "delete", mock_delete),
            patch("time.sleep") as mock_sleep,
            patch("logging.error") as mock_log,
        ):
            provider._stop_voice_iat()
            assert mock_delete.call_count == 3
            assert mock_sleep.call_count == 2
            assert mock_log.call_count == 3
            mock_log.assert_any_call(
                "UbtechASRProvider: _stop_voice_iat request failed (attempt 1): connection error"
            )
            mock_log.assert_any_call(
                "UbtechASRProvider: _stop_voice_iat request failed (attempt 2): connection error"
            )
            mock_log.assert_any_call(
                "UbtechASRProvider: _stop_voice_iat request failed (attempt 3): connection error"
            )

    def test_get_voice_iat_http_error_500(self, mock_requests):
        """Test _get_voice_iat retries on HTTP 500 errors."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        error_response = MagicMock()
        http_error_500 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=500)
        )
        error_response.raise_for_status.side_effect = http_error_500
        success_response = MagicMock()
        success_response.json.return_value = {"status": "ok"}
        success_response.raise_for_status.return_value = None

        mock_get = MagicMock(
            side_effect=[error_response, error_response, success_response]
        )
        with (
            patch.object(provider.session, "get", mock_get),
            patch("time.sleep") as mock_sleep,
        ):
            result = provider._get_voice_iat()
            assert result == {"status": "ok"}
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2

    def test_get_voice_iat_http_error_400(self, mock_requests):
        """Test _get_voice_iat handles non-retryable HTTP error."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        error_response = MagicMock()
        http_error_400 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=400)
        )
        error_response.raise_for_status.side_effect = http_error_400
        mock_get = MagicMock(return_value=error_response)
        with (
            patch.object(provider.session, "get", mock_get),
            patch("logging.error") as mock_log,
        ):
            result = provider._get_voice_iat()
            expected = {
                "code": -1,
                "message": str(http_error_400),
                "data": None,
                "status": "error",
            }
            assert result == expected
            mock_log.assert_called_once()

    def test_get_voice_iat_request_exception(self, mock_requests):
        """Test _get_voice_iat handles RequestException with retries."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        req_error = requests.RequestException("timeout")
        mock_get = MagicMock(side_effect=req_error)
        with (
            patch.object(provider.session, "get", mock_get),
            patch("time.sleep") as mock_sleep,
            patch("logging.error") as mock_log,
        ):
            result = provider._get_voice_iat()
            assert result["code"] == -1
            assert result["message"] == "timeout"
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2
            assert mock_log.call_count == 3

    def test_get_voice_iat_json_decode_error(self, mock_requests):
        """Test _get_voice_iat handles JSON decode error with retries."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get = MagicMock(return_value=mock_response)
        with (
            patch.object(provider.session, "get", mock_get),
            patch("time.sleep") as mock_sleep,
            patch("logging.error"),
        ):
            result = provider._get_voice_iat()
            assert result["code"] == -1
            assert "JSONDecodeError" in result["message"]
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2

    def test_get_voice_iat_data_string_cleaning(self, mock_requests):
        """Test cleaning of embedded JSON string in data field."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 0,
            "data": '{"text": {"ws": [{"cw": [{"w": "hi"}]}]}}\x00',
            "status": "idle",
        }
        mock_response.raise_for_status.return_value = None
        mock_get = MagicMock(return_value=mock_response)
        with patch.object(provider.session, "get", mock_get):
            result = provider._get_voice_iat()
            assert result["data"] == {"text": {"ws": [{"cw": [{"w": "hi"}]}]}}

    def test_get_voice_iat_data_string_invalid_json(self, mock_requests):
        """Test handling of invalid JSON in data field."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 0,
            "data": "invalid json",
            "status": "idle",
        }
        mock_response.raise_for_status.return_value = None
        mock_get = MagicMock(return_value=mock_response)
        with (
            patch.object(provider.session, "get", mock_get),
            patch("logging.error") as mock_log,
        ):
            result = provider._get_voice_iat()
            assert result["data"] == "invalid json"
            mock_log.assert_called_with(
                "UbtechASRProvider: Failed to decode JSON from data string: 'invalid json'"
            )

    def test_start_voice_iat_success(self, mock_requests):
        """Test _start_voice_iat success."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}
        mock_put = MagicMock(return_value=mock_response)
        with patch.object(provider.session, "put", mock_put):
            result = provider._start_voice_iat(12345)
            assert result is True
            mock_put.assert_called_once_with(
                "http://192.168.1.100:9090/v1/voice/iat",
                json={"text": "", "timestamp": 12345, "lang": "en"},
                timeout=3,
            )

    def test_start_voice_iat_failure(self, mock_requests):
        """Test _start_voice_iat failure (non-zero code)."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 1}
        mock_put = MagicMock(return_value=mock_response)
        with patch.object(provider.session, "put", mock_put):
            result = provider._start_voice_iat(12345)
            assert result is False

    def test_start_voice_iat_exception(self, mock_requests):
        """Test _start_voice_iat exception handling."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_put = MagicMock(side_effect=requests.RequestException("error"))
        with (
            patch.object(provider.session, "put", mock_put),
            patch("logging.error") as mock_log,
        ):
            result = provider._start_voice_iat(12345)
            assert result is False
            mock_log.assert_called_with(
                "UbtechASRProvider: _start_voice_iat request failed: error"
            )

    def test_set_robot_language_success(self, mock_requests):
        """Test _set_robot_language success."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_put = MagicMock()
        with patch.object(provider.session, "put", mock_put):
            provider._set_robot_language("en")
            mock_put.assert_called_once_with(
                "http://192.168.1.100:9090/v1/system/language",
                data=json.dumps({"language": "en"}),
                timeout=3,
            )

    def test_set_robot_language_exception(self, mock_requests):
        """Test _set_robot_language exception handling."""
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        mock_put = MagicMock(side_effect=requests.RequestException("error"))
        with (
            patch.object(provider.session, "put", mock_put),
            patch("logging.error") as mock_log,
        ):
            provider._set_robot_language("en")
            mock_log.assert_called_with(
                "UbtechASRProvider: Failed to set robot language: error"
            )
