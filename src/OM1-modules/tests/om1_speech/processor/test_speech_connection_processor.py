import argparse
import threading
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest

from om1_speech.interfaces import ASRProcessorInterface, AudioStreamInputInterface
from om1_speech.processor import ConnectionProcessor


class MockASRProcessor(ASRProcessorInterface):
    def __init__(self, args, callback=None):
        self.args = args
        self.callback = callback
        self.stopped = False

    def on_audio(self, audio: bytes) -> bytes:
        return audio

    def process_audio(self, audio_source: AudioStreamInputInterface) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


class MockAudioStreamInput(AudioStreamInputInterface):
    def __init__(self):
        self.stopped = False
        self.setup_called = False

    def handle_ws_incoming_message(self, connection_id: str, message: any) -> None:
        pass

    def setup_audio_stream(self) -> "AudioStreamInputInterface":
        self.setup_called = True
        return self

    def get_audio_chunk(self) -> Optional[bytes]:
        return None

    def stop(self) -> None:
        self.stopped = True

    def add(self, callback: callable) -> None:
        pass


@pytest.fixture
def mock_args():
    return argparse.Namespace(sample_rate=16000, channels=1)


@pytest.fixture
def connection_processor(mock_args):
    return ConnectionProcessor(mock_args, MockASRProcessor, MockAudioStreamInput)


@pytest.fixture
def mock_ws_server():
    server = MagicMock()
    server.register_connection_callback = Mock()
    server.register_message_callback = Mock()
    server.handle_response = Mock()
    return server


def test_connection_processor_initialization(connection_processor):
    """Test that ConnectionProcessor initializes correctly"""
    assert isinstance(connection_processor.args, argparse.Namespace)
    assert connection_processor.asr_processor_class == MockASRProcessor
    assert connection_processor.audio_stream_input_class == MockAudioStreamInput
    assert isinstance(connection_processor.asr_processors, dict)
    assert isinstance(connection_processor.audio_sources, dict)
    assert isinstance(connection_processor.processing_threads, dict)
    assert connection_processor.ws_server is None


def test_set_server(connection_processor, mock_ws_server):
    """Test setting the WebSocket server"""
    connection_processor.set_server(mock_ws_server)
    assert connection_processor.ws_server == mock_ws_server
    mock_ws_server.register_connection_callback.assert_called_once()


def test_handle_new_connection(connection_processor, mock_ws_server):
    """Test handling a new connection"""
    connection_processor.set_server(mock_ws_server)
    connection_id = "test_connection"

    # Handle new connection
    connection_processor.handle_new_connection(connection_id)

    # Verify ASR processor was created
    assert connection_id in connection_processor.asr_processors
    assert isinstance(
        connection_processor.asr_processors[connection_id], MockASRProcessor
    )

    # Verify audio source was created and setup
    assert connection_id in connection_processor.audio_sources
    assert isinstance(
        connection_processor.audio_sources[connection_id], MockAudioStreamInput
    )
    assert connection_processor.audio_sources[connection_id].setup_called

    # Verify thread was created and started
    assert connection_id in connection_processor.processing_threads
    assert isinstance(
        connection_processor.processing_threads[connection_id], threading.Thread
    )


def test_handle_connection_closed(connection_processor, mock_ws_server):
    """Test handling a connection being closed"""
    connection_processor.set_server(mock_ws_server)
    connection_id = "test_connection"

    # First create a connection
    connection_processor.handle_new_connection(connection_id)

    # Get references to created objects
    asr_processor = connection_processor.asr_processors[connection_id]
    audio_source = connection_processor.audio_sources[connection_id]

    # Handle connection closed
    connection_processor.handle_connection_closed(connection_id)

    # Verify objects were stopped and removed
    assert asr_processor.stopped
    assert audio_source.stopped
    assert connection_id not in connection_processor.asr_processors
    assert connection_id not in connection_processor.audio_sources
    assert connection_id not in connection_processor.processing_threads


def test_handle_connection_event(connection_processor, mock_ws_server):
    """Test handling different connection events"""
    connection_processor.set_server(mock_ws_server)
    connection_id = "test_connection"

    # Test connect event
    connection_processor.handle_connection_event("connect", connection_id)
    assert connection_id in connection_processor.asr_processors

    # Test disconnect event
    connection_processor.handle_connection_event("disconnect", connection_id)
    assert connection_id not in connection_processor.asr_processors


def test_stop(connection_processor, mock_ws_server):
    """Test stopping all connections"""
    connection_processor.set_server(mock_ws_server)

    # Create multiple connections
    connection_ids = ["conn1", "conn2", "conn3"]
    for conn_id in connection_ids:
        connection_processor.handle_new_connection(conn_id)

    # Stop all connections
    connection_processor.stop()

    # Verify all connections were closed
    assert len(connection_processor.asr_processors) == 0
    assert len(connection_processor.audio_sources) == 0
    assert len(connection_processor.processing_threads) == 0


@pytest.mark.parametrize(
    "connection_id",
    [
        "test_conn_1",
        "test_conn_2",
        "",  # Test empty string connection ID
        "123456789",  # Test numeric string connection ID
    ],
)
def test_connection_lifecycle(connection_processor, mock_ws_server, connection_id):
    """Test the complete lifecycle of a connection with different connection IDs"""
    connection_processor.set_server(mock_ws_server)

    # Create connection
    connection_processor.handle_new_connection(connection_id)
    assert connection_id in connection_processor.asr_processors
    assert connection_id in connection_processor.audio_sources
    assert connection_id in connection_processor.processing_threads

    # Close connection
    connection_processor.handle_connection_closed(connection_id)
    assert connection_id not in connection_processor.asr_processors
    assert connection_id not in connection_processor.audio_sources
    assert connection_id not in connection_processor.processing_threads
