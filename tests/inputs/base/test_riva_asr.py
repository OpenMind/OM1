import json
from unittest.mock import Mock, patch

import pytest

from inputs.plugins.riva_asr import RivaASRInput, RivaASRSensorConfig


@pytest.fixture
def mock_asr_provider():
    with patch("inputs.plugins.riva_asr.ASRProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_sleep_ticker():
    with patch("inputs.plugins.riva_asr.SleepTickerProvider") as mock:
        yield mock.return_value


@pytest.fixture
def asr_input(mock_asr_provider, mock_sleep_ticker):
    return RivaASRInput(config=RivaASRSensorConfig())


def test_init(asr_input, mock_asr_provider):
    assert asr_input.messages == []
    assert asr_input.message_buffer.empty()
    mock_asr_provider.start.assert_called_once()
    mock_asr_provider.register_message_callback.assert_called_once_with(
        asr_input._handle_asr_message
    )


def test_handle_asr_message(asr_input):
    test_message = json.dumps({"asr_reply": "test speech"})
    asr_input._handle_asr_message(test_message)
    assert asr_input.message_buffer.get_nowait() == "test speech"


def test_handle_asr_message_single_word(asr_input):
    test_message = json.dumps({"asr_reply": "test"})
    asr_input._handle_asr_message(test_message)
    assert asr_input.message_buffer.empty()


def test_handle_invalid_json(asr_input):
    invalid_json = "invalid json"
    asr_input._handle_asr_message(invalid_json)
    assert asr_input.message_buffer.empty()


@pytest.mark.asyncio
async def test_poll_with_message(asr_input):
    test_message = "test message"
    asr_input.message_buffer.put(test_message)
    result = await asr_input._poll()
    assert result == test_message


@pytest.mark.asyncio
async def test_poll_empty_queue(asr_input):
    result = await asr_input._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_conversion(asr_input):
    result = await asr_input._raw_to_text("test input")
    assert result.message == "test input"
    assert isinstance(result.timestamp, float)


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(asr_input, mock_sleep_ticker):
    await asr_input.raw_to_text("first message")
    assert asr_input.messages == ["first message"]

    await asr_input.raw_to_text("second message")
    assert asr_input.messages == ["first message second message"]


def test_formatted_latest_buffer(asr_input):
    asr_input.messages = ["test message"]
    result = asr_input.formatted_latest_buffer()
    assert "test message" in result
    assert asr_input.messages == []


def test_formatted_latest_buffer_empty(asr_input):
    assert asr_input.formatted_latest_buffer() is None
