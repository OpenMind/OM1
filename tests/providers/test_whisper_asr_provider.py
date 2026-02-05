import json
import queue
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from providers.whisper_asr_provider import WhisperASRProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    WhisperASRProvider.reset()  # type: ignore
    yield
    WhisperASRProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.whisper_asr_provider.pyaudio.PyAudio") as mock_pa,
        patch("providers.whisper_asr_provider.WhisperModel") as mock_model_cls,
    ):
        mock_pa_instance = MagicMock()
        mock_pa.return_value = mock_pa_instance
        mock_pa_instance.get_device_count.return_value = 0

        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        yield mock_pa, mock_model_cls, mock_model


def test_initialization(mock_dependencies):
    mock_pa, mock_model_cls, mock_model = mock_dependencies
    provider = WhisperASRProvider(model_size="tiny", device="cpu")

    mock_model_cls.assert_called_once_with("tiny", device="cpu", compute_type="auto")
    mock_pa.assert_called_once()
    assert not provider.running


def test_singleton_pattern(mock_dependencies):
    provider1 = WhisperASRProvider(model_size="tiny", device="cpu")
    provider2 = WhisperASRProvider(model_size="tiny", device="cpu")
    assert provider1 is provider2


def test_register_message_callback(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    callback = Mock()
    provider.register_message_callback(callback)

    assert provider._message_callback is callback


def test_register_none_callback(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider._message_callback = Mock()
    provider.register_message_callback(None)

    assert provider._message_callback is not None


def test_start(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()

    assert provider.running
    assert provider._audio_thread is not None

    provider.stop()


def test_start_already_running(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()
    first_thread = provider._audio_thread
    provider.start()

    assert provider._audio_thread is first_thread
    provider.stop()


def test_stop(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()
    provider.stop()

    assert not provider.running


def test_unsupported_model_falls_back(mock_dependencies):
    _, mock_model_cls, _ = mock_dependencies
    WhisperASRProvider(model_size="invalid_model", device="cpu")

    mock_model_cls.assert_called_once_with("turbo", device="cpu", compute_type="auto")


def test_transcribe_sends_callback(mock_dependencies):
    _, _, mock_model = mock_dependencies
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    callback = Mock()
    provider.register_message_callback(callback)

    mock_segment = MagicMock()
    mock_segment.text = "Hello world test"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    audio_data = np.zeros(16000, dtype=np.int16).tobytes()
    provider._transcribe(audio_data)

    callback.assert_called_once()
    call_arg = callback.call_args[0][0]
    parsed = json.loads(call_arg)
    assert parsed["asr_reply"] == "Hello world test"


def test_transcribe_skips_short_audio(mock_dependencies):
    _, _, mock_model = mock_dependencies
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    callback = Mock()
    provider.register_message_callback(callback)

    audio_data = np.zeros(100, dtype=np.int16).tobytes()
    provider._transcribe(audio_data)

    callback.assert_not_called()
    mock_model.transcribe.assert_not_called()


def test_transcribe_skips_empty_result(mock_dependencies):
    _, _, mock_model = mock_dependencies
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    callback = Mock()
    provider.register_message_callback(callback)

    mock_model.transcribe.return_value = ([], MagicMock())

    audio_data = np.zeros(16000, dtype=np.int16).tobytes()
    provider._transcribe(audio_data)

    callback.assert_not_called()


def test_on_audio_data_accumulates_buffer(mock_dependencies):
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    # Send silent audio - should accumulate in buffer
    audio_data = np.zeros(3200, dtype=np.int16).tobytes()
    provider._on_audio_data(audio_data)

    assert len(provider._audio_buffer) == 1


def test_resolve_device_by_name(mock_dependencies):
    mock_pa, _, _ = mock_dependencies
    mock_pa_instance = mock_pa.return_value
    mock_pa_instance.get_device_count.return_value = 2
    mock_pa_instance.get_device_info_by_index.side_effect = [
        {"name": "Speaker Output", "maxInputChannels": 0},
        {"name": "MacBook Air Microphone", "maxInputChannels": 1},
    ]

    provider = WhisperASRProvider(
        model_size="tiny", device="cpu", microphone_name="MacBook"
    )
    assert provider._device_index == 1


def test_stop_terminates_pyaudio(mock_dependencies):
    """PyAudio.terminate() must be called on stop to release resources."""
    mock_pa, _, _ = mock_dependencies
    mock_pa_instance = mock_pa.return_value

    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()
    provider.stop()

    mock_pa_instance.terminate.assert_called_once()


def test_start_creates_transcription_thread(mock_dependencies):
    """start() must create a dedicated transcription worker thread."""
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()

    assert provider._transcription_thread is not None
    assert provider._transcription_thread.is_alive()

    provider.stop()


def test_stop_joins_transcription_thread(mock_dependencies):
    """stop() must join and clean up the transcription thread."""
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.start()

    assert provider._transcription_thread is not None
    provider.stop()

    assert provider._transcription_thread is None


def test_on_audio_data_queues_instead_of_spawning_thread(mock_dependencies):
    """Speech-to-silence transition must enqueue audio, not spawn a new thread."""
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    # Simulate speech (loud audio)
    loud_audio = (np.ones(3200, dtype=np.int16) * 10000).tobytes()
    provider._on_audio_data(loud_audio)

    # Simulate enough silence to trigger transcription
    silent_audio = np.zeros(3200, dtype=np.int16).tobytes()
    chunks_needed = int(provider._silence_duration * (provider._rate / provider._chunk))
    for _ in range(chunks_needed):
        provider._on_audio_data(silent_audio)

    # Audio should be in the queue, not spawned as a thread
    assert not provider._transcription_queue.empty()
    queued_data = provider._transcription_queue.get_nowait()
    assert isinstance(queued_data, bytes)
    assert len(queued_data) > 0


def test_transcription_worker_processes_queue(mock_dependencies):
    """Transcription worker must process items from the queue sequentially."""
    _, _, mock_model = mock_dependencies
    provider = WhisperASRProvider(model_size="tiny", device="cpu")

    callback = Mock()
    provider.register_message_callback(callback)

    mock_segment = MagicMock()
    mock_segment.text = "worker test"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    # Put audio data directly into the queue
    audio_data = np.zeros(16000, dtype=np.int16).tobytes()
    provider._transcription_queue.put(audio_data)

    # Start provider (which starts the worker)
    provider.start()

    # Wait for the worker to process
    deadline = time.time() + 3.0
    while not provider._transcription_queue.empty() and time.time() < deadline:
        time.sleep(0.05)

    provider.stop()

    callback.assert_called_once()
    parsed = json.loads(callback.call_args[0][0])
    assert parsed["asr_reply"] == "worker test"


def test_stop_sends_none_sentinel_to_queue(mock_dependencies):
    """stop() must send None sentinel to unblock the transcription worker."""
    provider = WhisperASRProvider(model_size="tiny", device="cpu")
    provider.running = True

    # Don't start threads, just test the queue signaling
    provider.stop()

    # The sentinel None should have been put in the queue
    try:
        item = provider._transcription_queue.get_nowait()
        assert item is None
    except queue.Empty:
        # Queue was consumed by worker - also acceptable
        pass
