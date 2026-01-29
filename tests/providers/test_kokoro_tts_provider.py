import sys
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

# Mock om1_speech module
mock_om1_speech = MagicMock()
mock_om1_speech.AudioOutputLiveStream = MagicMock()
sys.modules["om1_speech"] = mock_om1_speech

# Import after mocking
from providers.kokoro_tts_provider import KokoroTTSProvider  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    KokoroTTSProvider.reset()  # type: ignore
    mock_om1_speech.AudioOutputLiveStream.reset_mock()
    mock_om1_speech.AudioOutputLiveStream.return_value = MagicMock()
    yield
    KokoroTTSProvider.reset()  # type: ignore


def test_initialization_with_defaults():
    """Test provider initializes with default parameters."""
    provider = KokoroTTSProvider()

    assert provider.api_key is None
    assert provider.running is False
    assert provider._voice_id == "af_bella"
    assert provider._model_id == "kokoro"
    assert provider._output_format == "pcm"
    assert provider._enable_tts_interrupt is False


def test_initialization_with_custom_parameters():
    """Test provider initializes with custom parameters."""
    custom_params = {
        "url": "http://custom.url:9999/v1",
        "api_key": "test_api_key",
        "voice_id": "custom_voice",
        "model_id": "custom_model",
        "output_format": "wav",
        "rate": 48000,
        "enable_tts_interrupt": True,
    }

    provider = KokoroTTSProvider(**custom_params)

    assert provider.api_key == "test_api_key"
    assert provider._voice_id == "custom_voice"
    assert provider._model_id == "custom_model"
    assert provider._output_format == "wav"
    assert provider._enable_tts_interrupt is True


def test_audio_stream_initialization():
    """Test that AudioOutputLiveStream is initialized with correct parameters."""
    with patch("providers.kokoro_tts_provider.AudioOutputLiveStream") as mock_stream:
        KokoroTTSProvider(
            url="http://test.url",
            api_key="test_key",
            voice_id="test_voice",
            model_id="test_model",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
        )

        mock_stream.assert_called_once_with(
            url="http://test.url",
            tts_model="test_model",
            tts_voice="test_voice",
            response_format="wav",
            rate=48000,
            api_key="test_key",
            enable_tts_interrupt=True,
        )


def test_configure_no_restart_needed_when_not_running():
    """Test configure doesn't call stop when provider is not running and no changes."""
    provider = KokoroTTSProvider()
    provider.running = False

    with patch.object(provider, "stop") as mock_stop:
        provider.configure()
        mock_stop.assert_not_called()


def test_configure_restart_needed_when_running():
    """Test configure calls stop when running and parameters change."""
    provider = KokoroTTSProvider(voice_id="original_voice")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(voice_id="new_voice")
        mock_stop.assert_called_once()


def test_configure_restart_needed_url_change():
    """Test restart is triggered when URL changes."""
    original_url = "http://original.url"
    new_url = "http://new.url"

    provider = KokoroTTSProvider(url=original_url)
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(url=new_url)
        mock_stop.assert_called_once()


def test_configure_restart_needed_api_key_change():
    """Test restart is triggered when API key changes."""
    provider = KokoroTTSProvider(api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_voice_id_change():
    """Test restart is triggered when voice ID changes."""
    provider = KokoroTTSProvider(voice_id="original_voice")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(voice_id="new_voice")
        mock_stop.assert_called_once()


def test_configure_restart_needed_model_id_change():
    """Test restart is triggered when model ID changes."""
    provider = KokoroTTSProvider(model_id="original_model")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(model_id="new_model")
        mock_stop.assert_called_once()


def test_configure_restart_needed_output_format_change():
    """Test restart is triggered when output format changes."""
    provider = KokoroTTSProvider(output_format="pcm")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(output_format="wav")
        mock_stop.assert_called_once()


def test_configure_restart_needed_interrupt_setting_change():
    """Test restart is triggered when TTS interrupt setting changes."""
    provider = KokoroTTSProvider(enable_tts_interrupt=False)
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(enable_tts_interrupt=True)
        mock_stop.assert_called_once()


def test_configure_no_restart_when_same_parameters():
    """Test no restart when all parameters remain the same."""
    url = "http://test.url"
    api_key = "test_key"
    voice_id = "test_voice"
    model_id = "test_model"
    output_format = "pcm"
    enable_tts_interrupt = False

    mock_audio_stream = MagicMock()
    mock_audio_stream._url = url
    mock_om1_speech.AudioOutputLiveStream.return_value = mock_audio_stream

    provider = KokoroTTSProvider(
        url=url,
        api_key=api_key,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        enable_tts_interrupt=enable_tts_interrupt,
    )
    provider.running = True
    provider._audio_stream._url = url

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(
            url=url,
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        mock_stop.assert_not_called()


def test_configure_creates_new_audio_stream():
    """Test configure creates new AudioOutputLiveStream when restart needed."""
    with patch(
        "providers.kokoro_tts_provider.AudioOutputLiveStream"
    ) as mock_stream_class:
        mock_stream_instance = MagicMock()
        mock_stream_class.return_value = mock_stream_instance

        provider = KokoroTTSProvider(voice_id="original_voice")
        provider.running = True

        provider.configure(voice_id="new_voice")

        # Should be called twice: once in __init__, once in configure
        assert mock_stream_class.call_count == 2
        # Check that start was called on the new stream
        mock_stream_instance.start.assert_called()


def test_register_tts_state_callback():
    """Test registering TTS state callback."""
    provider = KokoroTTSProvider()
    callback = MagicMock()

    provider.register_tts_state_callback(callback)

    cast(
        MagicMock, provider._audio_stream.set_tts_state_callback
    ).assert_called_once_with(callback)


def test_register_tts_state_callback_none():
    """Test registering None callback doesn't call set_tts_state_callback."""
    provider = KokoroTTSProvider()

    provider.register_tts_state_callback(None)

    cast(MagicMock, provider._audio_stream.set_tts_state_callback).assert_not_called()


def test_create_pending_message():
    """Test creating pending message with text input."""
    provider = KokoroTTSProvider(
        voice_id="test_voice", model_id="test_model", output_format="wav"
    )

    with patch("providers.kokoro_tts_provider.logging") as mock_logging:
        result = provider.create_pending_message("Hello world")

        expected = {
            "text": "Hello world",
            "voice_id": "test_voice",
            "model_id": "test_model",
            "output_format": "wav",
        }

        assert result == expected
        mock_logging.info.assert_called_once_with("audio_stream: Hello world")


def test_add_pending_message_string():
    """Test adding pending message with string input."""
    provider = KokoroTTSProvider()
    provider.running = True

    with patch.object(provider, "create_pending_message") as mock_create:
        mock_create.return_value = {"text": "test", "voice_id": "af_bella"}

        provider.add_pending_message("test message")

        mock_create.assert_called_once_with("test message")
        cast(MagicMock, provider._audio_stream.add_request).assert_called_once()


def test_add_pending_message_dict():
    """Test adding pending message with dict input."""
    provider = KokoroTTSProvider()
    provider.running = True

    message_dict = {"text": "test", "voice_id": "custom_voice"}

    with patch("providers.kokoro_tts_provider.logging") as mock_logging:
        provider.add_pending_message(message_dict)

        cast(MagicMock, provider._audio_stream.add_request).assert_called_once_with(
            message_dict
        )
        mock_logging.info.assert_called()


def test_add_pending_message_not_running():
    """Test adding pending message when provider is not running."""
    provider = KokoroTTSProvider()
    provider.running = False

    with patch("providers.kokoro_tts_provider.logging") as mock_logging:
        provider.add_pending_message("test message")

        cast(MagicMock, provider._audio_stream.add_request).assert_not_called()
        mock_logging.warning.assert_called_once_with(
            "TTS provider is not running. Call start() before adding messages."
        )


def test_get_pending_message_count():
    """Test getting pending message count."""
    provider = KokoroTTSProvider()
    cast(MagicMock, provider._audio_stream._pending_requests.qsize).return_value = 5

    count = provider.get_pending_message_count()

    assert count == 5
    cast(MagicMock, provider._audio_stream._pending_requests.qsize).assert_called_once()


def test_start_when_not_running():
    """Test starting the provider when it's not running."""
    provider = KokoroTTSProvider()
    provider.running = False

    provider.start()

    assert provider.running is True
    cast(MagicMock, provider._audio_stream.start).assert_called_once()


def test_start_when_already_running():
    """Test starting the provider when it's already running."""
    provider = KokoroTTSProvider()
    provider.running = True

    with patch("providers.kokoro_tts_provider.logging") as mock_logging:
        provider.start()

        mock_logging.warning.assert_called_once_with(
            "Eleven Labs TTS provider is already running"
        )


def test_stop_when_running():
    """Test stopping the provider when it's running."""
    provider = KokoroTTSProvider()
    provider.running = True

    provider.stop()

    assert provider.running is False
    cast(MagicMock, provider._audio_stream.stop).assert_called_once()


def test_stop_when_not_running():
    """Test stopping the provider when it's not running."""
    provider = KokoroTTSProvider()
    provider.running = False

    with patch("providers.kokoro_tts_provider.logging") as mock_logging:
        provider.stop()

        mock_logging.warning.assert_called_once_with(
            "Eleven Labs TTS provider is not running"
        )


def test_start_stop_integration():
    """Test complete start and stop cycle."""
    provider = KokoroTTSProvider()

    # Initial state
    assert provider.running is False

    # Start
    provider.start()
    assert provider.running is True
    cast(MagicMock, provider._audio_stream.start).assert_called_once()

    # Stop
    provider.stop()
    assert provider.running is False
    cast(MagicMock, provider._audio_stream.stop).assert_called_once()


def test_singleton_behavior():
    """Test that KokoroTTSProvider behaves as a singleton."""
    provider1 = KokoroTTSProvider()
    provider2 = KokoroTTSProvider()

    assert provider1 is provider2


def test_configure_updates_internal_state():
    """Test that configure updates internal state variables."""
    provider = KokoroTTSProvider()

    provider.configure(
        voice_id="new_voice",
        model_id="new_model",
        output_format="wav",
        enable_tts_interrupt=True,
    )

    assert provider._voice_id == "new_voice"
    assert provider._model_id == "new_model"
    assert provider._output_format == "wav"
    assert provider._enable_tts_interrupt is True
