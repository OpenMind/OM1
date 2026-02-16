import sys
from unittest.mock import MagicMock, patch

# Mock ubtech modules before any imports
sys.modules["ubtech"] = MagicMock()
sys.modules["ubtech.ubtechapi"] = MagicMock()


import pytest  # noqa: E402

from inputs.base import Message  # noqa: E402
from inputs.plugins.ubtech_camera_vlm_input import (  # noqa: E402
    UbtechCameraVLMInput,
    UbtechCameraVLMSensorConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
    ):
        config = UbtechCameraVLMSensorConfig()
        sensor = UbtechCameraVLMInput(config=config)

        assert hasattr(sensor, "messages")


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.ubtech_camera_vlm_input.IOProvider"),
        patch("inputs.plugins.ubtech_camera_vlm_input.UbtechVLMProvider"),
    ):
        config = UbtechCameraVLMSensorConfig(
            robot_ip="192.168.1.100", base_url="wss://test.com"
        )
        sensor = UbtechCameraVLMInput(config=config)

        assert sensor.config.robot_ip == "192.168.1.100"
        assert sensor.config.base_url == "wss://test.com"


def test_handle_vlm_message_with_valid_json():
    """Test _handle_vlm_message with valid VLM JSON (lines 83-90)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    # Valid JSON with vlm_reply
    json_msg = '{"vlm_reply": "This is a test response"}'
    input_obj._handle_vlm_message(json_msg)

    assert input_obj.message_buffer.qsize() == 1
    assert input_obj.message_buffer.get() == "This is a test response"


def test_handle_vlm_message_with_invalid_json():
    """Test _handle_vlm_message with invalid JSON (line 89-90)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    # Invalid JSON should not crash
    input_obj._handle_vlm_message("not a json")

    assert input_obj.message_buffer.qsize() == 0


@pytest.mark.asyncio
async def test_poll_returns_message():
    """Test _poll returns message from buffer (line 107)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    # Add message to buffer
    input_obj.message_buffer.put("Test message from buffer")

    result = await input_obj._poll()
    assert result == "Test message from buffer"


@pytest.mark.asyncio
async def test_poll_empty_buffer():
    """Test _poll returns None when buffer is empty (lines 108-109)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    # Buffer is empty
    result = await input_obj._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_io_provider():
    """Test raw_to_text appends to messages (lines 145-151)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    initial_count = len(input_obj.messages)

    # Test with valid input
    await input_obj.raw_to_text("Valid message")
    assert len(input_obj.messages) == initial_count + 1

    # Test with None input
    await input_obj.raw_to_text(None)
    assert len(input_obj.messages) == initial_count + 1  # Should not increase


@pytest.mark.asyncio
async def test_raw_to_text_internal_with_none():
    """Test _raw_to_text returns None when input is None (line 129)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    result = await input_obj._raw_to_text(None)
    assert result is None


def test_formatted_latest_buffer_empty_messages():
    """Test formatted_latest_buffer with no messages (lines 168-169)."""
    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    result = input_obj.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages (lines 170-185)."""
    import time

    config = UbtechCameraVLMSensorConfig()
    input_obj = UbtechCameraVLMInput(config)

    # Add messages
    msg1 = Message(timestamp=time.time(), message="First message")
    msg2 = Message(timestamp=time.time(), message="Second message")
    input_obj.messages.append(msg1)
    input_obj.messages.append(msg2)

    result = input_obj.formatted_latest_buffer()

    assert result is not None
    assert "Second message" in result
    assert "INPUT:" in result
    assert "// START" in result
    assert "// END" in result
    assert len(input_obj.messages) == 0  # Should be cleared
