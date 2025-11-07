import threading
from queue import Queue
from unittest.mock import Mock

import pytest

from om1_utils.ws import Client


@pytest.fixture
def mock_websocket():
    """Fixture to create a mock websocket connection"""
    mock_ws = Mock()
    # Mock the required websocket methods
    mock_ws.recv.return_value = "test message"
    mock_ws.send.return_value = None
    mock_ws.close.return_value = None
    # Prevent actual socket operations
    mock_ws.socket = Mock()
    return mock_ws


@pytest.fixture
def client():
    """Fixture to create a client instance"""
    return Client("ws://test.com")


def test_client_initialization(client):
    """Test client initialization with default values"""
    assert client.url == "ws://test.com"
    assert client.running is True
    assert client.connected is False
    assert client.websocket is None
    assert client.message_callback is None
    assert isinstance(client.message_queue, Queue)
    assert client.receiver_thread is None
    assert client.sender_thread is None


def test_register_message_callback(client):
    """Test callback registration"""

    def callback(message):
        pass

    client.register_message_callback(callback)
    assert client.message_callback == callback


def test_send_message_when_connected(client):
    """Test message sending when client is connected"""
    client.connected = True
    test_message = "Hello, WebSocket!"

    client.send_message(test_message)

    assert client.message_queue.qsize() == 1
    assert client.message_queue.get() == test_message


def test_send_message_when_disconnected(client):
    """Test message sending when client is disconnected"""
    client.connected = False
    test_message = "Hello, WebSocket!"

    client.send_message(test_message)

    assert client.message_queue.empty()


def test_format_message_short(client):
    """Test message formatting with short message"""
    short_message = "Short test message"
    formatted = client.format_message(short_message)
    assert formatted == short_message


def test_format_message_long(client):
    """Test message formatting with long message"""
    long_message = "x" * 300
    formatted = client.format_message(long_message, max_length=100)
    assert len(formatted) <= 100
    assert "..." in formatted


def test_stop_client(client, mock_websocket):
    """Test client stop functionality"""
    client.websocket = mock_websocket
    client.connected = True

    client.stop()

    assert client.running is False
    assert client.connected is False
    assert client.message_queue.empty()
    mock_websocket.close.assert_called_once()


def test_receive_messages_with_callback(client, mock_websocket):
    """Test message receiving with callback"""
    received_messages = []

    def callback(message):
        received_messages.append(message)

    client.websocket = mock_websocket
    client.connected = True
    client.register_message_callback(callback)

    # Start receiver thread
    receiver_thread = threading.Thread(target=client._receive_messages)
    receiver_thread.daemon = True
    receiver_thread.start()

    # Let the thread run briefly
    threading.Event().wait(0.1)

    # Stop the client
    client.running = False
    receiver_thread.join(timeout=1)

    assert len(received_messages) > 0
    assert received_messages[0] == "test message"
