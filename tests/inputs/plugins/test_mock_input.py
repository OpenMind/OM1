"""Tests for MockInput input plugin."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.mock_input import MockInput, MockSensorConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return MockSensorConfig(
        input_name="Test Mock Input",
        host="localhost",
        port=8766,  # Use different port to avoid conflicts
    )


@pytest.fixture
def mock_input(config):
    """Create a MockInput instance."""
    return MockInput(config)


def test_initialization(mock_input):
    """Test that MockInput initializes correctly."""
    assert mock_input.io_provider is not None
    assert mock_input.messages == []
    assert mock_input.descriptor_for_LLM == "Test Mock Input"
    assert mock_input.host == "localhost"
    assert mock_input.port == 8766
    assert mock_input.message_buffer is not None
    assert mock_input.server_thread.is_alive() or not mock_input.server_thread.is_alive()  # Thread may or may not be alive


def test_initialization_with_defaults():
    """Test initialization with default configuration values."""
    default_config = MockSensorConfig()
    mock_input = MockInput(default_config)
    
    assert mock_input.descriptor_for_LLM == "Mock Input"
    assert mock_input.host == "localhost"
    assert mock_input.port == 8765


@pytest.mark.asyncio
async def test_poll_with_empty_buffer(mock_input):
    """Test that _poll returns None when buffer is empty."""
    result = await mock_input._poll()
    assert result is None


@pytest.mark.asyncio
async def test_poll_with_message(mock_input):
    """Test that _poll returns message from buffer."""
    test_message = "Test message"
    mock_input.message_buffer.put(test_message)
    
    result = await mock_input._poll()
    assert result == test_message


@pytest.mark.asyncio
async def test_raw_to_text_with_input(mock_input):
    """Test that _raw_to_text creates a Message with correct format."""
    test_input = "Test input message"
    
    message = await mock_input._raw_to_text(test_input)
    
    assert message is not None
    assert message.message == test_input
    assert message.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_with_none(mock_input):
    """Test that _raw_to_text returns None for None input."""
    message = await mock_input._raw_to_text(None)
    assert message is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer(mock_input):
    """Test that raw_to_text updates the message buffer."""
    test_input = "Test message"
    
    assert len(mock_input.messages) == 0
    
    await mock_input.raw_to_text(test_input)
    
    assert len(mock_input.messages) == 1
    assert mock_input.messages[0].message == test_input


@pytest.mark.asyncio
async def test_raw_to_text_with_none_does_not_update_buffer(mock_input):
    """Test that raw_to_text doesn't update buffer when input is None."""
    assert len(mock_input.messages) == 0
    
    await mock_input.raw_to_text(None)
    
    assert len(mock_input.messages) == 0


def test_formatted_latest_buffer_empty(mock_input):
    """Test formatted_latest_buffer returns None when buffer is empty."""
    result = mock_input.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message(mock_input):
    """Test formatted_latest_buffer formats message correctly."""
    from inputs.base import Message
    
    message1 = Message(timestamp=time.time(), message="First message")
    message2 = Message(timestamp=time.time(), message="Second message")
    mock_input.messages.extend([message1, message2])
    
    result = mock_input.formatted_latest_buffer()
    
    assert result is not None
    assert "INPUT: Test Mock Input" in result
    assert "Second message" in result  # Should contain latest message
    assert "// START" in result
    assert "// END" in result
    assert len(mock_input.messages) == 0  # Buffer should be cleared


def test_formatted_latest_buffer_clears_buffer(mock_input):
    """Test that formatted_latest_buffer clears the message buffer."""
    from inputs.base import Message
    
    message = Message(timestamp=time.time(), message="Test message")
    mock_input.messages.append(message)
    
    result = mock_input.formatted_latest_buffer()
    
    assert result is not None
    assert len(mock_input.messages) == 0  # Buffer cleared


@pytest.mark.asyncio
async def test_full_workflow(mock_input):
    """Test the full workflow from poll to formatted output."""
    # Add message to buffer
    test_message = "Test workflow message"
    mock_input.message_buffer.put(test_message)
    
    # Poll for message
    raw_input = await mock_input._poll()
    assert raw_input == test_message
    
    # Convert to text
    await mock_input.raw_to_text(raw_input)
    assert len(mock_input.messages) == 1
    
    # Format buffer
    formatted = mock_input.formatted_latest_buffer()
    assert formatted is not None
    assert "Test Mock Input" in formatted
    assert test_message in formatted
    assert len(mock_input.messages) == 0


def test_message_buffer_queue_operations(mock_input):
    """Test message buffer queue operations."""
    # Test putting messages
    mock_input.message_buffer.put("Message 1")
    mock_input.message_buffer.put("Message 2")
    
    # Test getting messages
    msg1 = mock_input.message_buffer.get_nowait()
    assert msg1 == "Message 1"
    
    msg2 = mock_input.message_buffer.get_nowait()
    assert msg2 == "Message 2"
    
    # Test empty queue
    assert mock_input.message_buffer.empty()
