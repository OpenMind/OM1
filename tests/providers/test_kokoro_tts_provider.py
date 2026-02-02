import logging
from unittest.mock import MagicMock, patch

import pytest

from providers.kokoro_tts_provider import KokoroTTSProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test to ensure test isolation."""
    KokoroTTSProvider.reset()  # type: ignore
    yield
    KokoroTTSProvider.reset()  # type: ignore


@pytest.fixture
def mock_audio_stream():
    """Fixture for mocked AudioOutputLiveStream."""
    with patch("providers.kokoro_tts_provider.AudioOutputLiveStream") as mock:
        mock_instance = MagicMock()
        mock_instance._url = "http://127.0.0.1:8880/v1"
        mock_instance._pending_requests = MagicMock()
        mock_instance._pending_requests.qsize.return_value = 0
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def provider(mock_audio_stream):
    """Fixture for KokoroTTSProvider with mocked audio stream."""
    provider = KokoroTTSProvider()
    return provider


class TestKokoroTTSProviderInit:
    """Test initialization of KokoroTTSProvider."""

    def test_init_default_parameters(self, mock_audio_stream):
        """Test initialization with default parameters."""
        provider = KokoroTTSProvider()

        assert provider.api_key is None
        assert provider.running is False
        assert provider._voice_id == "af_bella"
        assert provider._model_id == "kokoro"
        assert provider._output_format == "pcm"
        assert provider._enable_tts_interrupt is False

        mock_audio_stream.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            tts_model="kokoro",
            tts_voice="af_bella",
            response_format="pcm",
            rate=24000,
            api_key=None,
            enable_tts_interrupt=False,
        )

    def test_init_custom_parameters(self, mock_audio_stream):
        """Test initialization with custom parameters."""
        provider = KokoroTTSProvider(
            url="http://custom:9000/v1",
            api_key="test-key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
        )

        assert provider.api_key == "test-key"
        assert provider.running is False
        assert provider._voice_id == "custom_voice"
        assert provider._model_id == "custom_model"
        assert provider._output_format == "wav"
        assert provider._enable_tts_interrupt is True

        mock_audio_stream.assert_called_once_with(
            url="http://custom:9000/v1",
            tts_model="custom_model",
            tts_voice="custom_voice",
            response_format="wav",
            rate=48000,
            api_key="test-key",
            enable_tts_interrupt=True,
        )


class TestKokoroTTSProviderConfigure:
    """Test configuration of KokoroTTSProvider."""

    def test_configure_no_changes(self, provider):
        """Test configure when no changes are needed."""
        provider.running = True
        initial_stream = provider._audio_stream

        provider.configure()

        # Should not restart or create new stream
        assert provider._audio_stream is initial_stream

    def test_configure_with_changes_while_stopped(self, provider, mock_audio_stream):
        """Test configure with changes while provider is stopped."""
        provider.configure(
            voice_id="new_voice",
            model_id="new_model",
        )

        assert provider._voice_id == "new_voice"
        assert provider._model_id == "new_model"
        provider._audio_stream.start.assert_called_once()

    def test_configure_with_changes_while_running(self, provider, mock_audio_stream):
        """Test configure with changes while provider is running."""
        provider.running = True

        provider.configure(
            voice_id="new_voice",
            output_format="wav",
        )

        assert provider._voice_id == "new_voice"
        assert provider._output_format == "wav"
        # Should stop and recreate stream
        assert provider.running is False
        provider._audio_stream.start.assert_called()


class TestKokoroTTSProviderCallbacks:
    """Test callback registration."""

    def test_register_tts_state_callback(self, provider):
        """Test registering a TTS state callback."""
        callback = MagicMock()

        provider.register_tts_state_callback(callback)

        provider._audio_stream.set_tts_state_callback.assert_called_once_with(callback)

    def test_register_tts_state_callback_none(self, provider):
        """Test registering None callback does nothing."""
        provider.register_tts_state_callback(None)

        provider._audio_stream.set_tts_state_callback.assert_not_called()


class TestKokoroTTSProviderMessages:
    """Test message handling."""

    def test_create_pending_message(self, provider):
        """Test creating a pending message."""
        result = provider.create_pending_message("Hello world")

        assert result == {
            "text": "Hello world",
            "voice_id": "af_bella",
            "model_id": "kokoro",
            "output_format": "pcm",
        }

    def test_add_pending_message_as_string(self, provider):
        """Test adding a pending message as string."""
        provider.running = True

        provider.add_pending_message("Test message")

        provider._audio_stream.add_request.assert_called_once()
        call_args = provider._audio_stream.add_request.call_args[0][0]
        assert call_args["text"] == "Test message"
        assert call_args["voice_id"] == "af_bella"

    def test_add_pending_message_as_dict(self, provider):
        """Test adding a pending message as dict."""
        provider.running = True
        message = {
            "text": "Test message",
            "voice_id": "custom_voice",
            "model_id": "custom_model",
            "output_format": "wav",
        }

        provider.add_pending_message(message)

        provider._audio_stream.add_request.assert_called_once_with(message)

    def test_add_pending_message_not_running(self, provider, caplog):
        """Test adding a pending message when provider is not running."""
        provider.running = False

        with caplog.at_level(logging.WARNING):
            provider.add_pending_message("Test message")

        provider._audio_stream.add_request.assert_not_called()
        assert "TTS provider is not running" in caplog.text

    def test_get_pending_message_count(self, provider):
        """Test getting pending message count."""
        provider._audio_stream._pending_requests.qsize.return_value = 5

        count = provider.get_pending_message_count()

        assert count == 5


class TestKokoroTTSProviderLifecycle:
    """Test start and stop lifecycle."""

    def test_start(self, provider):
        """Test starting the provider."""
        provider.running = False

        provider.start()

        assert provider.running is True
        provider._audio_stream.start.assert_called_once()

    def test_start_already_running(self, provider, caplog):
        """Test starting when already running."""
        provider.running = True

        with caplog.at_level(logging.WARNING):
            provider.start()

        assert "already running" in caplog.text

    def test_stop(self, provider):
        """Test stopping the provider."""
        provider.running = True

        provider.stop()

        assert provider.running is False
        provider._audio_stream.stop.assert_called_once()

    def test_stop_not_running(self, provider, caplog):
        """Test stopping when not running."""
        provider.running = False

        with caplog.at_level(logging.WARNING):
            provider.stop()

        assert "is not running" in caplog.text
