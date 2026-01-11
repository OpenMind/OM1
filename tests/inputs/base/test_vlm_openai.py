from unittest.mock import Mock, patch

import pytest

from inputs.plugins.vlm_openai import Message, VLMOpenAI, VLMOpenAIConfig


@pytest.fixture
def mock_vlm_provider():
    with patch("inputs.plugins.vlm_openai.VLMOpenAIProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def vlm_input(mock_vlm_provider):
    return VLMOpenAI(config=VLMOpenAIConfig(api_key="test_api_key"))


def test_missing_api_key():
    with pytest.raises(ValueError):
        VLMOpenAI(config=VLMOpenAIConfig(api_key=""))


def test_handle_vlm_message(vlm_input):
    image_content = "image content"
    mock_chat_completion = Mock()
    mock_chat_completion.choices = [Mock(message=Mock(content=image_content))]
    vlm_input._handle_vlm_message(mock_chat_completion)
    assert vlm_input.message_buffer.get_nowait() == image_content


@pytest.mark.asyncio
async def test_poll_with_message(vlm_input):
    test_message = "test message"
    vlm_input.message_buffer.put(test_message)
    result = await vlm_input._poll()
    assert result == test_message


@pytest.mark.asyncio
async def test_poll_empty_queue(vlm_input):
    result = await vlm_input._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_conversion(vlm_input):
    result = await vlm_input._raw_to_text("test input")
    assert result.message == "test input"


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(vlm_input):
    await vlm_input.raw_to_text("first message")
    assert len(vlm_input.messages) == 1
    assert vlm_input.messages[0].message == "first message"

    await vlm_input.raw_to_text("second message")
    assert len(vlm_input.messages) == 2
    assert vlm_input.messages[1].message == "second message"


def test_formatted_latest_buffer(vlm_input):
    vlm_input.messages = [Message(message="test message", timestamp=123.456)]
    result = vlm_input.formatted_latest_buffer()
    assert "test message" in result
    assert vlm_input.messages == []


def test_formatted_latest_buffer_empty(vlm_input):
    assert vlm_input.formatted_latest_buffer() is None
