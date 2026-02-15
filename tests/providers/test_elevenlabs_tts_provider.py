import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_external_modules(monkeypatch):
    """Mock external modules before importing the provider module."""
    mock_om1_speech = MagicMock()
    mock_om1_speech.AudioOutputStream = MagicMock()

    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_default_output_device_info.return_value = {
        "name": "Mock Speaker",
        "index": 0,
    }
    mock_pyaudio.PyAudio.return_value = mock_instance

    monkeypatch.setitem(sys.modules, "om1_speech", mock_om1_speech)
    monkeypatch.setitem(sys.modules, "pyaudio", mock_pyaudio)

    return {
        "om1_speech": mock_om1_speech,
        "pyaudio": mock_pyaudio,
    }


@pytest.fixture
def provider_module(mock_external_modules):
    """Import the provider module after mocking external deps."""
    sys.modules.pop("providers.elevenlabs_tts_provider", None)
    return importlib.import_module("providers.elevenlabs_tts_provider")


@pytest.fixture(autouse=True)
def reset_singleton(provider_module):
    """Reset singleton instances between tests."""
    provider_module.ElevenLabsTTSProvider.reset()  # type: ignore
    yield
    provider_module.ElevenLabsTTSProvider.reset()  # type: ignore


def test_configure_no_restart_needed_when_not_running(provider_module):
    """Test configure doesn't call stop when provider is not running."""
    provider = provider_module.ElevenLabsTTSProvider()
    provider.running = False

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="same_key")
        mock_stop.assert_not_called()


def test_configure_restart_needed_when_running(provider_module):
    """Test configure calls stop when running and parameters change."""
    provider = provider_module.ElevenLabsTTSProvider(api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_url_change(provider_module):
    """Test restart is triggered when URL changes."""
    original_url = "https://original.api.com"
    new_url = "https://new.api.com"

    provider = provider_module.ElevenLabsTTSProvider(url=original_url)
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(url=new_url)
        mock_stop.assert_called_once()


def test_configure_restart_needed_api_key_change(provider_module):
    """Test restart is triggered when API key changes."""
    provider = provider_module.ElevenLabsTTSProvider(api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_elevenlabs_api_key_change(provider_module):
    """Test restart is triggered when ElevenLabs API key changes."""
    provider = provider_module.ElevenLabsTTSProvider(elevenlabs_api_key="original_key")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(elevenlabs_api_key="new_key")
        mock_stop.assert_called_once()


def test_configure_restart_needed_voice_id_change(provider_module):
    """Test restart is triggered when voice ID changes."""
    provider = provider_module.ElevenLabsTTSProvider(voice_id="original_voice")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(voice_id="new_voice")
        mock_stop.assert_called_once()


def test_configure_restart_needed_model_id_change(provider_module):
    """Test restart is triggered when model ID changes."""
    provider = provider_module.ElevenLabsTTSProvider(model_id="original_model")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(model_id="new_model")
        mock_stop.assert_called_once()


def test_configure_restart_needed_output_format_change(provider_module):
    """Test restart is triggered when output format changes."""
    provider = provider_module.ElevenLabsTTSProvider(output_format="pcm_16000")
    provider.running = True

    with patch.object(provider, "stop") as mock_stop:
        provider.configure(output_format="pcm_44100")
        mock_stop.assert_called_once()


def test_configure_no_restart_when_same_parameters(
    provider_module, mock_external_modules
):
    """Test no restart when all parameters remain the same."""
    url = "https://api.openmind.org/api/core/elevenlabs/tts"
    api_key = "same_key"
    elevenlabs_key = "same_elevenlabs_key"
    voice_id = "same_voice"
    model_id = "same_model"
    output_format = "same_format"

    mock_audio_stream = MagicMock()
    mock_audio_stream._url = url
    mock_external_modules["om1_speech"].AudioOutputStream.return_value = (
        mock_audio_stream
    )

    provider = provider_module.ElevenLabsTTSProvider(
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


def test_start_stop(provider_module):
    """Test start and stop functionality."""
    provider = provider_module.ElevenLabsTTSProvider(url="test_url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False
