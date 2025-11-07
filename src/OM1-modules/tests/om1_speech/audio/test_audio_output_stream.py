import base64
import json
import time
from queue import Queue
from unittest.mock import Mock, patch

import pytest

from om1_speech.audio.audio_output_stream import AudioOutputStream


@pytest.fixture
def mock_pyaudio():
    with patch("pyaudio.PyAudio") as mock:
        # Setup default device info
        device_info = {
            "name": "Test Device",
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100,
            "index": 0,
        }

        # Mock device count and default device info
        mock.return_value.get_device_count.return_value = 2
        mock.return_value.get_device_info_by_index.return_value = device_info
        mock.return_value.get_default_output_device_info.return_value = device_info

        yield mock


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.Popen") as mock:
        # Mock process and communicate
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.poll.return_value = 0
        mock.return_value = mock_process
        yield mock


@pytest.fixture
def mock_ffplay():
    with patch("om1_speech.audio.audio_output_stream.is_installed") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_requests():
    with patch("requests.post") as mock:
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": base64.b64encode(b"test_audio_data").decode("utf-8")
        }
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def audio_output(mock_pyaudio, mock_requests, mock_subprocess, mock_ffplay):
    stream = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)
    yield stream
    stream.stop()


def test_initialization(audio_output, mock_pyaudio):
    """Test AudioOutputStream initialization"""
    assert audio_output._rate == 16000
    assert audio_output._url == "http://test-tts-server/tts"
    assert audio_output.running is True
    assert isinstance(audio_output._pending_requests, Queue)


def test_tts_callback(mock_pyaudio):
    """Test TTS state callback"""
    callback_state = None

    def tts_callback(state):
        nonlocal callback_state
        callback_state = state

    stream = AudioOutputStream(
        "http://test-tts-server/tts", tts_state_callback=tts_callback
    )

    stream._tts_callback(True)
    assert callback_state is True

    stream._tts_callback(False)
    assert callback_state is False

    stream.stop()


def test_audio_processing(audio_output, mock_requests, mock_subprocess):
    """Test audio processing flow"""
    # Create a flag for callback verification
    callback_called = False

    def tts_callback(state):
        nonlocal callback_called
        callback_called = state

    audio_output.set_tts_state_callback(tts_callback)

    # Start processing
    audio_output.start()

    # Add test input
    test_text = "Hello, world!"
    audio_output.add_request({"text": test_text})

    # Wait a bit for processing
    time.sleep(0.5)

    # Verify request was made
    mock_requests.assert_called_with(
        "http://test-tts-server/tts",
        data=json.dumps({"text": test_text}),
        headers={"Content-Type": "application/json"},
        timeout=(5, 15),
    )

    # Verify ffplay was called with correct parameters
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[1]["args"]
    assert "ffplay" in args[0]
    assert "-autoexit" in args


def test_error_handling(audio_output, mock_requests):
    """Test error handling in audio processing"""
    # Make the request fail
    mock_requests.return_value.status_code = 500

    # Start processing
    audio_output.start()

    # Add test input
    audio_output.add_request({"text": "Test error handling"})

    # Wait a bit for processing
    time.sleep(0.1)

    # Verify stream is still running
    assert audio_output.running is True


def test_stop(audio_output):
    """Test stopping the audio output stream"""
    audio_output.start()
    audio_output.stop()

    assert audio_output.running is False


def test_empty_queue_handling(audio_output):
    """Test handling of empty queue"""
    audio_output.start()

    # Wait a bit with empty queue
    time.sleep(0.1)

    # Verify stream is still running
    assert audio_output.running is True


@pytest.mark.parametrize("rate", [8000, 16000, 44100])
def test_different_sample_rates(mock_pyaudio, rate):
    """Test initialization with different sample rates"""
    stream = AudioOutputStream("http://test-tts-server/tts", rate=rate)

    # Verify rate was stored correctly
    assert stream._rate == rate

    stream.stop()


def test_add_multiple_items(audio_output):
    """Test adding multiple items to the queue"""
    items = [{"text": "Test 1"}, {"text": "Test 2"}, {"text": "Test 3"}]

    for item in items:
        audio_output.add_request(item)

    # Verify items were added to queue
    assert audio_output._pending_requests.qsize() == len(items)

    # Verify queue contents
    received_items = []
    while not audio_output._pending_requests.empty():
        received_items.append(audio_output._pending_requests.get())

    assert received_items == items


def test_subprocess_error(audio_output, mock_subprocess, mock_requests):
    """Test handling of subprocess errors"""
    # Make subprocess return error code
    mock_subprocess.return_value.poll.return_value = 1

    # Start processing
    audio_output.start()

    # Add test input
    audio_output.add_request({"text": "Test subprocess error"})

    # Wait a bit for processing
    time.sleep(0.1)

    # Verify stream is still running despite the error
    assert audio_output.running is True


def test_create_silence_audio(audio_output):
    """Test silent audio generation"""
    # Test default duration (500ms)
    silence_audio = audio_output._create_silence_audio()
    silence_bytes = base64.b64decode(silence_audio)

    # Calculate expected size for 500ms at 16000 Hz (16-bit samples)
    expected_samples = int(16000 * 500 / 1000)
    expected_size = expected_samples * 2  # 16-bit = 2 bytes per sample

    assert len(silence_bytes) == expected_size
    assert silence_bytes == b"\x00" * expected_size

    # Test custom duration (100ms)
    silence_audio_100ms = audio_output._create_silence_audio(100)
    silence_bytes_100ms = base64.b64decode(silence_audio_100ms)

    expected_samples_100ms = int(16000 * 100 / 1000)
    expected_size_100ms = expected_samples_100ms * 2

    assert len(silence_bytes_100ms) == expected_size_100ms


def test_audio_prefix_for_bluetooth(
    mock_pyaudio, mock_subprocess, mock_requests, mock_ffplay
):
    """Test that audio has silence prefix to prevent Bluetooth clipping"""
    audio_output = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)

    # Start processing
    audio_output.start()

    # Add test input
    test_audio_data = base64.b64encode(b"original_audio_data").decode("utf-8")
    mock_requests.return_value.json.return_value = {"response": test_audio_data}

    audio_output.add_request({"text": "Test audio with prefix"})

    # Wait for processing
    time.sleep(0.5)

    # Verify ffplay was called
    assert mock_subprocess.called

    # Get the audio data that was passed to ffplay via communicate
    communicate_input = mock_subprocess.return_value.communicate.call_args[1]["input"]

    # The audio should be longer than the original due to the silence prefix
    original_audio = base64.b64decode(test_audio_data)
    assert len(communicate_input) > len(original_audio)

    audio_output.stop()


def test_keepalive_timing(audio_output):
    """Test keepalive timing logic"""
    # Initialize last audio time
    audio_output._last_audio_time = time.time() - 65  # 65 seconds ago

    # Check if keepalive should trigger
    current_time = time.time()
    should_trigger = current_time - audio_output._last_audio_time >= 60

    assert should_trigger is True

    # Update time and check again
    audio_output._last_audio_time = time.time()
    should_trigger = current_time - audio_output._last_audio_time >= 60

    assert should_trigger is False


def test_keepalive_thread_behavior(
    mock_pyaudio, mock_subprocess, mock_requests, mock_ffplay
):
    """Test keepalive thread behavior over time"""
    audio_output = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)

    # Set last audio time to trigger keepalive
    audio_output._last_audio_time = time.time() - 70

    # Start the stream
    audio_output.start()

    # Wait briefly for keepalive thread to process
    time.sleep(0.2)

    # Verify keepalive sound was played
    assert mock_subprocess.called

    audio_output.stop()


def test_write_audio_updates_last_audio_time(
    mock_pyaudio, mock_subprocess, mock_requests, mock_ffplay
):
    """Test that _write_audio updates the last audio time"""
    audio_output = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)

    initial_time = audio_output._last_audio_time

    # Simulate writing audio
    test_audio = base64.b64encode(b"test_audio").decode("utf-8")
    audio_output._write_audio(test_audio)

    # Verify last audio time was updated
    assert audio_output._last_audio_time > initial_time

    audio_output.stop()


def test_keepalive_vs_regular_audio_callbacks(
    mock_pyaudio, mock_subprocess, mock_requests, mock_ffplay
):
    """Test that keepalive sounds don't trigger TTS callbacks but regular audio does"""
    audio_output = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)

    callback_states = []

    def tts_callback(state):
        callback_states.append(state)

    audio_output.set_tts_state_callback(tts_callback)

    # Test keepalive sound (should not trigger callbacks)
    silence_audio = audio_output._create_silence_audio(100)
    audio_output._write_audio_bytes(silence_audio, is_keepalive=True)

    # No callbacks should be recorded
    assert len(callback_states) == 0

    # Test regular audio (should trigger callbacks)
    regular_audio = base64.b64encode(b"regular_audio").decode("utf-8")
    audio_output._write_audio_bytes(regular_audio, is_keepalive=False)

    # Should have start and end callbacks
    assert len(callback_states) == 2
    assert callback_states[0] is True  # Start callback
    assert callback_states[1] is False  # End callback

    audio_output.stop()
