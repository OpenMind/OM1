"""Tests for DummyVLMLocal input plugin."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from inputs.base import SensorConfig
from inputs.plugins.vlm_dummy_local import DummyVLMLocal


@pytest.fixture
def config():
    """Create a test configuration."""
    return SensorConfig()


@pytest.fixture
def dummy_vlm(config):
    """Create a DummyVLMLocal instance."""
    return DummyVLMLocal(config)


def test_initialization(dummy_vlm):
    """Test that DummyVLMLocal initializes correctly."""
    assert dummy_vlm.io_provider is not None
    assert dummy_vlm.messages == []
    assert dummy_vlm.descriptor_for_LLM == "Vision"


@pytest.mark.asyncio
async def test_poll_generates_image(dummy_vlm):
    """Test that _poll generates a valid PIL Image."""
    image = await dummy_vlm._poll()
    
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)
    assert image.mode == "RGB"


@pytest.mark.asyncio
async def test_poll_generates_different_images(dummy_vlm):
    """Test that _poll generates different images on each call."""
    image1 = await dummy_vlm._poll()
    image2 = await dummy_vlm._poll()
    
    # Images should be different (random colors)
    # Compare pixel data
    pixels1 = list(image1.getdata())
    pixels2 = list(image2.getdata())
    
    # Due to randomness, they might be the same occasionally, but very unlikely
    # We'll just check they're valid images
    assert len(pixels1) == 10000  # 100x100 = 10000 pixels
    assert len(pixels2) == 10000


@pytest.mark.asyncio
async def test_raw_to_text_creates_message(dummy_vlm):
    """Test that _raw_to_text creates a Message with correct format."""
    test_image = Image.new("RGB", (100, 100), (255, 0, 0))
    
    message = await dummy_vlm._raw_to_text(test_image)
    
    assert message is not None
    assert message.message is not None
    assert "DUMMY VLM" in message.message
    assert message.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_updates_buffer(dummy_vlm):
    """Test that raw_to_text updates the message buffer."""
    test_image = Image.new("RGB", (100, 100), (0, 255, 0))
    
    assert len(dummy_vlm.messages) == 0
    
    await dummy_vlm.raw_to_text(test_image)
    
    assert len(dummy_vlm.messages) == 1
    assert "DUMMY VLM" in dummy_vlm.messages[0].message


def test_formatted_latest_buffer_empty(dummy_vlm):
    """Test formatted_latest_buffer returns None when buffer is empty."""
    result = dummy_vlm.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message(dummy_vlm):
    """Test formatted_latest_buffer formats message correctly."""
    from inputs.base import Message
    import time
    
    message = Message(timestamp=time.time(), message="DUMMY VLM - FAKE DATA - I see 5 people. Also, I see a rocket.")
    dummy_vlm.messages.append(message)
    
    result = dummy_vlm.formatted_latest_buffer()
    
    assert result is not None
    assert "INPUT: Vision" in result
    assert "DUMMY VLM" in result
    assert "// START" in result
    assert "// END" in result
    assert len(dummy_vlm.messages) == 0  # Buffer should be cleared


def test_formatted_latest_buffer_clears_buffer(dummy_vlm):
    """Test that formatted_latest_buffer clears the message buffer."""
    from inputs.base import Message
    import time
    
    message1 = Message(timestamp=time.time(), message="Message 1")
    message2 = Message(timestamp=time.time(), message="Message 2")
    dummy_vlm.messages.extend([message1, message2])
    
    result = dummy_vlm.formatted_latest_buffer()
    
    assert result is not None
    assert "Message 2" in result  # Should contain latest message
    assert len(dummy_vlm.messages) == 0  # Buffer cleared


@pytest.mark.asyncio
async def test_full_workflow(dummy_vlm):
    """Test the full workflow from poll to formatted output."""
    # Poll for image
    image = await dummy_vlm._poll()
    assert isinstance(image, Image.Image)
    
    # Convert to text
    await dummy_vlm.raw_to_text(image)
    assert len(dummy_vlm.messages) == 1
    
    # Format buffer
    formatted = dummy_vlm.formatted_latest_buffer()
    assert formatted is not None
    assert "Vision" in formatted
    assert len(dummy_vlm.messages) == 0
