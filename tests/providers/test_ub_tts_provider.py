from unittest.mock import MagicMock, patch

from providers.ub_tts_provider import UbTtsProvider


class TestUbTtsProviderInitialization:
    """Tests for UbTtsProvider initialization."""

    def test_initialization_sets_url(self):
        """Test that initialization correctly sets the TTS URL."""
        provider = UbTtsProvider("http://localhost:8080/tts")
        assert provider.tts_url == "http://localhost:8080/tts"

    def test_initialization_sets_headers(self):
        """Test that initialization sets correct headers."""
        provider = UbTtsProvider("http://localhost:8080/tts")
        assert provider.headers == {"Content-Type": "application/json"}


class TestUbTtsProviderSpeak:
    """Tests for the speak() method."""

    def test_speak_success(self):
        """Test successful TTS speak request."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "providers.ub_tts_provider.requests.put", return_value=mock_response
        ) as mock_put:
            result = provider.speak("Hello world")

            assert result is True
            mock_put.assert_called_once()
            call_kwargs = mock_put.call_args[1]
            assert call_kwargs["url"] == "http://localhost:8080/tts"
            assert call_kwargs["timeout"] == 5

    def test_speak_with_parameters(self):
        """Test speak with custom interrupt and timestamp parameters."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "providers.ub_tts_provider.requests.put", return_value=mock_response
        ) as mock_put:
            result = provider.speak("Hello", interrupt=False, timestamp=12345)

            assert result is True
            call_data = mock_put.call_args[1]["data"]
            assert '"interrupt": false' in call_data
            assert '"timestamp": 12345' in call_data

    def test_speak_failure_non_zero_code(self):
        """Test speak returns False when response code is non-zero."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 1, "error": "TTS busy"}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "providers.ub_tts_provider.requests.put", return_value=mock_response
        ):
            result = provider.speak("Hello")
            assert result is False

    def test_speak_request_exception(self):
        """Test speak handles request exceptions gracefully."""
        import requests

        provider = UbTtsProvider("http://localhost:8080/tts")

        with patch(
            "providers.ub_tts_provider.requests.put",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            result = provider.speak("Hello")
            assert result is False

    def test_speak_timeout_exception(self):
        """Test speak handles timeout exceptions gracefully."""
        import requests

        provider = UbTtsProvider("http://localhost:8080/tts")

        with patch(
            "providers.ub_tts_provider.requests.put",
            side_effect=requests.exceptions.Timeout("Request timed out"),
        ):
            result = provider.speak("Hello")
            assert result is False


class TestUbTtsProviderGetStatus:
    """Tests for the get_tts_status() method."""

    def test_get_status_success(self):
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

    def test_get_status_idle(self):
        """Test status returns idle when TTS is not active."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "status": "idle"}

        with patch(
            "providers.ub_tts_provider.requests.get", return_value=mock_response
        ):
            result = provider.get_tts_status(0)
            assert result == "idle"

    def test_get_status_non_zero_code(self):
        """Test status returns error when response code is non-zero."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 1}

        with patch(
            "providers.ub_tts_provider.requests.get", return_value=mock_response
        ):
            result = provider.get_tts_status(12345)
            assert result == "error"

    def test_get_status_request_exception(self):
        """Test status returns error on request exception."""
        import requests

        provider = UbTtsProvider("http://localhost:8080/tts")

        with patch(
            "providers.ub_tts_provider.requests.get",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            result = provider.get_tts_status(12345)
            assert result == "error"

    def test_get_status_missing_status_field(self):
        """Test status returns error when status field is missing."""
        provider = UbTtsProvider("http://localhost:8080/tts")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}  # No status field

        with patch(
            "providers.ub_tts_provider.requests.get", return_value=mock_response
        ):
            result = provider.get_tts_status(12345)
            assert result == "error"
