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
from providers.singleton import singleton  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    singleton.instances = {}
    yield


def test_init_with_default_parameters():
    """Test initialization with default parameters."""
    provider = ElevenLabsTTSProvider()

    assert provider.api_key is None
    assert provider.elevenlabs_api_key is None
    assert provider._voice_id == "JBFqnCBsd6RMkjVDRZzb"
    assert provider._model_id == "eleven_flash_v2_5"
    assert provider._output_format == "mp3_44100_128"
    assert provider.running is False

    mock_om1_speech.AudioOutputStream.assert_called_with(
        url="https://api.openmind.org/api/core/elevenlabs/tts", headers=None
    )


def test_init_with_custom_parameters():
    """Test initialization with custom parameters."""
    custom_url = "https://custom.api.com/tts"
    custom_api_key = "test_api_key"
    custom_elevenlabs_key = "test_elevenlabs_key"
    custom_voice_id = "custom_voice"
    custom_model_id = "custom_model"
    custom_format = "mp3_22050_64"

    provider = ElevenLabsTTSProvider(
        url=custom_url,
        api_key=custom_api_key,
        elevenlabs_api_key=custom_elevenlabs_key,
        voice_id=custom_voice_id,
        model_id=custom_model_id,
        output_format=custom_format,
    )

    assert provider.api_key == custom_api_key
    assert provider.elevenlabs_api_key == custom_elevenlabs_key
    assert provider._voice_id == custom_voice_id
    assert provider._model_id == custom_model_id
    assert provider._output_format == custom_format
    assert provider.running is False

    mock_om1_speech.AudioOutputStream.assert_called_with(
        url=custom_url, headers={"x-api-key": custom_api_key}
    )


def test_init_without_api_key():
    """Test initialization without API key creates headers as None."""
    ElevenLabsTTSProvider(url="test_url")

    mock_om1_speech.AudioOutputStream.assert_called_with(url="test_url", headers=None)


def test_configure_with_default_parameters():
    """Test configure method with default parameters."""
    provider = ElevenLabsTTSProvider()

    mock_om1_speech.AudioOutputStream.reset_mock()

    mock_audio_stream = MagicMock()
    mock_om1_speech.AudioOutputStream.return_value = mock_audio_stream

    provider.configure()

    assert provider.api_key is None
    assert provider.elevenlabs_api_key is None
    assert provider._voice_id == "JBFqnCBsd6RMkjVDRZzb"
    assert provider._model_id == "eleven_flash_v2_5"
    assert provider._output_format == "mp3_44100_128"

    mock_om1_speech.AudioOutputStream.assert_called_with(
        url="https://api.openmind.org/api/core/elevenlabs/tts", headers=None
    )
    mock_audio_stream.start.assert_called_once()


def test_configure_with_custom_parameters():
    """Test configure method with custom parameters."""
    provider = ElevenLabsTTSProvider()

    mock_om1_speech.AudioOutputStream.reset_mock()

    mock_audio_stream = MagicMock()
    mock_om1_speech.AudioOutputStream.return_value = mock_audio_stream

    custom_url = "https://custom.api.com/tts"
    custom_api_key = "new_api_key"
    custom_elevenlabs_key = "new_elevenlabs_key"
    custom_voice_id = "new_voice"
    custom_model_id = "new_model"
    custom_format = "mp3_16000_32"

    provider.configure(
        url=custom_url,
        api_key=custom_api_key,
        elevenlabs_api_key=custom_elevenlabs_key,
        voice_id=custom_voice_id,
        model_id=custom_model_id,
        output_format=custom_format,
    )

    assert provider.api_key == custom_api_key
    assert provider.elevenlabs_api_key == custom_elevenlabs_key
    assert provider._voice_id == custom_voice_id
    assert provider._model_id == custom_model_id
    assert provider._output_format == custom_format

    mock_om1_speech.AudioOutputStream.assert_called_with(
        url=custom_url, headers={"x-api-key": custom_api_key}
    )
    mock_audio_stream.start.assert_called_once()


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


def test_configure_with_none_api_keys():
    """Test configure with None API keys creates headers as None."""
    provider = ElevenLabsTTSProvider()

    mock_om1_speech.AudioOutputStream.reset_mock()

    mock_audio_stream = MagicMock()
    mock_om1_speech.AudioOutputStream.return_value = mock_audio_stream

    provider.configure(api_key=None, elevenlabs_api_key=None)

    mock_om1_speech.AudioOutputStream.assert_called_with(
        url="https://api.openmind.org/api/core/elevenlabs/tts", headers=None
    )


def test_configure_updates_audio_stream_reference():
    """Test that configure creates a new audio stream instance."""
    provider = ElevenLabsTTSProvider()
    original_audio_stream = provider._audio_stream

    mock_om1_speech.AudioOutputStream.reset_mock()

    mock_audio_stream = MagicMock()
    mock_om1_speech.AudioOutputStream.return_value = mock_audio_stream

    provider.configure(api_key="new_key")

    assert mock_om1_speech.AudioOutputStream.call_count == 1
    assert provider._audio_stream == mock_audio_stream
    assert provider._audio_stream is not original_audio_stream


def test_start_stop():
    """Test start and stop functionality."""
    provider = ElevenLabsTTSProvider(url="test_url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False
