import sys
from unittest.mock import MagicMock, patch

import pytest

mock_om1_speech = MagicMock()
mock_om1_speech.AudioOutputStream = MagicMock()
sys.modules["om1_speech"] = mock_om1_speech

mock_pyaudio = MagicMock()
mock_pyaudio.PyAudio = MagicMock()
mock_instance = MagicMock()
mock_instance.get_default_output_device_info.return_value = {
    "name": "Mock Speaker",
    "index": 0,
}
mock_pyaudio.PyAudio.return_value = mock_instance
sys.modules["pyaudio"] = mock_pyaudio

# Import after mocking
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ElevenLabsTTSProvider.reset()  # type: ignore
    yield
    ElevenLabsTTSProvider.reset()  # type: ignore


def test_configure_no_restart_needed_when_not_running():
    """Test configure doesn't call stop when provider is not running."""
    provider = ElevenLabsTTSProvider()
    provider.running = False

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="same_key")
        mock_stop.assert_not_called()


def test_configure_restart_needed_when_running():
    """Test configure calls stop when running and parameters change."""
    provider = ElevenLabsTTSProvider(api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_url_change():
    """Test restart is triggered when URL changes."""
    original_url = "https://original.api.com"
    new_url = "https://new.api.com"

    provider = ElevenLabsTTSProvider(url=original_url)
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(url=new_url)
        mock_stop.assert_called_once()


def test_configure_restart_needed_api_key_change():
    """Test restart is triggered when API key changes."""
    provider = ElevenLabsTTSProvider(api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_elevenlabs_api_key_change():
    """Test restart is triggered when ElevenLabs API key changes."""
    provider = ElevenLabsTTSProvider(elevenlabs_api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(elevenlabs_api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_voice_id_change():
    """Test restart is triggered when voice ID changes."""
    provider = ElevenLabsTTSProvider(voice_id="original_voice")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(voice_id="new_voice")
        mock_stop.assert_called_once()


def test_configure_restart_needed_model_id_change():
    """Test restart is triggered when model ID changes."""
    provider = ElevenLabsTTSProvider(model_id="original_model")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(model_id="new_model")
        mock_stop.assert_called_once()


def test_configure_restart_needed_output_format_change():
    """Test restart is triggered when output format changes."""
    provider = ElevenLabsTTSProvider(output_format="mp3_44100_128")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(output_format="mp3_22050_64")
        mock_stop.assert_called_once()


def test_configure_no_restart_when_same_parameters():
    """Test no restart when all parameters remain the same."""
    url = "https://api.openmind.org/api/core/elevenlabs/tts"
    api_key = "same_key"
    elevenlabs_key = "same_elevenlabs_key"
    voice_id = "same_voice"
    model_id = "same_model"
    output_format = "same_format"

    mock_audio_stream = MagicMock()
    mock_audio_stream._url = url
    mock_om1_speech.AudioOutputStream.return_value = mock_audio_stream

    provider = ElevenLabsTTSProvider(
        url=url,
        api_key=api_key,
        elevenlabs_api_key=elevenlabs_key,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    provider.running = True

    provider._audio_stream._url = url

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(
            url=url,
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        mock_stop.assert_not_called()


def test_start_stop():
    """Test start and stop functionality."""
    provider = ElevenLabsTTSProvider(url="test_url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False
