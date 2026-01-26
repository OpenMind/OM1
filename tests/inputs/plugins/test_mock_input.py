"""Tests for MockInput input plugin."""

import asyncio
import time
from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.mock_input import MockInput, MockSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        assert sensor.messages == []
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.host == "localhost"
        assert sensor.port == 8765
        assert sensor.descriptor_for_LLM == "Mock Input"


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig(input_name="Custom Mock", host="0.0.0.0", port=9000)
        sensor = MockInput(config=config)

        assert sensor.descriptor_for_LLM == "Custom Mock"
        assert sensor.host == "0.0.0.0"
        assert sensor.port == 9000


@pytest.mark.asyncio
async def test_poll_with_message_in_buffer():
    """Test _poll when there's a message in buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.message_buffer.put("Test message")

        result = await sensor._poll()

        assert result == "Test message"


@pytest.mark.asyncio
async def test_poll_with_empty_buffer():
    """Test _poll when buffer is empty."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.time.time", return_value=1234.0),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._raw_to_text("Test message")

        assert result is not None
        assert result.timestamp == 1234.0
        assert result.message == "Test message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer():
    """Test that raw_to_text updates the message buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        
        assert len(sensor.messages) == 0
        
        await sensor.raw_to_text("Test message")
        
        assert len(sensor.messages) == 1
        assert sensor.messages[0].message == "Test message"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_does_not_update_buffer():
    """Test that raw_to_text doesn't update buffer when input is None."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        
        assert len(sensor.messages) == 0
        
        await sensor.raw_to_text(None)
        
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Message 1"),
            Message(timestamp=1001.0, message="Message 2"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Message 2" in result  # Should contain latest message
        assert "INPUT: Mock Input" in result
        assert "// START" in result
        assert "// END" in result
        sensor.io_provider.add_input.assert_called()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


def test_formatted_latest_buffer_clears_buffer():
    """Test that formatted_latest_buffer clears the message buffer."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.io_provider = MagicMock()
        
        message = Message(timestamp=time.time(), message="Test message")
        sensor.messages.append(message)
        
        result = sensor.formatted_latest_buffer()
        
        assert result is not None
        assert len(sensor.messages) == 0  # Buffer cleared


@pytest.mark.asyncio
async def test_full_workflow():
    """Test the full workflow from poll to formatted output."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
        patch("inputs.plugins.mock_input.asyncio.sleep", new=AsyncMock()),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        sensor.io_provider = MagicMock()
        
        # Add message to buffer
        test_message = "Test workflow message"
        sensor.message_buffer.put(test_message)
        
        # Poll for message
        raw_input = await sensor._poll()
        assert raw_input == test_message
        
        # Convert to text
        await sensor.raw_to_text(raw_input)
        assert len(sensor.messages) == 1
        
        # Format buffer
        formatted = sensor.formatted_latest_buffer()
        assert formatted is not None
        assert "Mock Input" in formatted
        assert test_message in formatted
        assert len(sensor.messages) == 0


def test_message_buffer_queue_operations():
    """Test message buffer queue operations."""
    with (
        patch("inputs.plugins.mock_input.IOProvider"),
        patch.object(MockInput, "_start_server_thread"),
    ):
        config = MockSensorConfig()
        sensor = MockInput(config=config)
        
        # Test putting messages
        sensor.message_buffer.put("Message 1")
        sensor.message_buffer.put("Message 2")
        
        # Test getting messages
        msg1 = sensor.message_buffer.get_nowait()
        assert msg1 == "Message 1"
        
        msg2 = sensor.message_buffer.get_nowait()
        assert msg2 == "Message 2"
        
        # Test empty queue
        assert sensor.message_buffer.empty()
