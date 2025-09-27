import sys
from unittest.mock import MagicMock

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


def test_start_stop():
    provider = ElevenLabsTTSProvider(url="test_url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False
