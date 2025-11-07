import argparse
import threading
from unittest.mock import ANY, Mock

import pytest

from om1_vlm import ConnectionProcessor


@pytest.fixture
def mock_components():
    """Fixture providing mock VLM processor and video stream components."""
    mock_vlm_processor = Mock()
    mock_video_stream = Mock()
    return mock_vlm_processor, mock_video_stream


@pytest.fixture
def processor(mock_components):
    """Fixture providing configured ConnectionProcessor instance."""
    mock_vlm_processor, mock_video_stream = mock_components
    args = argparse.Namespace()

    processor = ConnectionProcessor(
        args,
        vlm_processor_class=lambda *args, **kwargs: mock_vlm_processor,
        video_stream_input_class=lambda: mock_video_stream,
    )

    # Set up mock WebSocket server
    mock_ws_server = Mock()
    processor.set_server(mock_ws_server)

    # Add mock server to processor for test access
    processor.mock_ws_server = mock_ws_server

    return processor


def test_set_server(processor):
    """Test WebSocket server initialization."""
    # Verify that connection callback was registered
    processor.mock_ws_server.register_connection_callback.assert_called_once()


def test_handle_new_connection(processor, mock_components):
    """Test handling of new WebSocket connections."""
    mock_vlm_processor, mock_video_stream = mock_components
    connection_id = "test_conn_1"

    # Handle new connection
    processor.handle_new_connection(connection_id)

    # Verify VLM processor was created
    assert connection_id in processor.vlm_processors

    # Verify video source was created and configured
    assert connection_id in processor.video_sources
    mock_video_stream.setup_video_stream.assert_called_once_with(
        processor.args, mock_vlm_processor.on_video
    )

    # Verify message callback was registered
    processor.mock_ws_server.register_message_callback.assert_called_once_with(
        connection_id, ANY
    )

    # Verify processing thread was created and started
    assert connection_id in processor.processing_threads
    assert isinstance(processor.processing_threads[connection_id], threading.Thread)


def test_handle_connection_closed(processor, mock_components):
    """Test cleanup when connection closes."""
    mock_vlm_processor, mock_video_stream = mock_components
    connection_id = "test_conn_1"

    # First create a connection
    processor.handle_new_connection(connection_id)

    # Then close it
    processor.handle_connection_closed(connection_id)

    # Verify VLM processor was stopped and removed
    mock_vlm_processor.stop.assert_called_once()
    assert connection_id not in processor.vlm_processors

    # Verify video source was stopped and removed
    mock_video_stream.stop.assert_called_once()
    assert connection_id not in processor.video_sources

    # Verify processing thread was removed
    assert connection_id not in processor.processing_threads


def test_handle_connection_event(processor, mock_components):
    """Test connection event handling."""
    connection_id = "test_conn_1"

    # Test connect event
    processor.handle_connection_event("connect", connection_id)
    assert connection_id in processor.vlm_processors

    # Test disconnect event
    processor.handle_connection_event("disconnect", connection_id)
    assert connection_id not in processor.vlm_processors


def test_stop(processor, mock_components):
    """Test stopping all connections."""
    mock_vlm_processor, mock_video_stream = mock_components

    # Create multiple connections
    connection_ids = ["conn_1", "conn_2", "conn_3"]
    for conn_id in connection_ids:
        processor.handle_new_connection(conn_id)

    # Stop all connections
    processor.stop()

    # Verify all resources were cleaned up
    assert len(processor.vlm_processors) == 0
    assert len(processor.video_sources) == 0
    assert len(processor.processing_threads) == 0

    # Verify stop was called on all processors and sources
    assert mock_vlm_processor.stop.call_count == len(connection_ids)
    assert mock_video_stream.stop.call_count == len(connection_ids)


@pytest.mark.parametrize(
    "connection_id,expected_vlm_count",
    [
        ("test_1", 1),
        ("test_2", 1),
        ("test_with_special_chars_#@!", 1),
    ],
)
def test_connection_creation_parameterized(
    processor, mock_components, connection_id, expected_vlm_count
):
    """Test connection creation with different connection IDs."""
    processor.handle_new_connection(connection_id)
    assert len(processor.vlm_processors) == expected_vlm_count
    assert connection_id in processor.vlm_processors
