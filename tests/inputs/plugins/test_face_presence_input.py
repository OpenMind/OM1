import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from inputs.plugins.face_presence_input import FacePresence, FacePresenceConfig


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.face_presence_input.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_initialization_creates_providers_and_buffers(mock_io_provider):
    config = FacePresenceConfig()
    mock_provider_instance = Mock()

    mock_provider_constructor = Mock(return_value=mock_provider_instance)

    with patch(
        "inputs.plugins.face_presence_input.FacePresenceProvider",
        new=mock_provider_constructor,
    ):
        with patch(
            "inputs.plugins.face_presence_input.IOProvider",
            return_value=mock_io_provider,
        ):
            instance = FacePresence(config=config)

    mock_provider_constructor.assert_called_once()
    mock_provider_instance.start.assert_called_once()
    mock_provider_instance.register_message_callback.assert_called_once()

    assert instance.io_provider is not None
    assert mock_io_provider is not None

    assert hasattr(instance, "messages")
    assert isinstance(instance.messages, deque)
    assert instance.messages.maxlen == 300

    assert hasattr(instance, "message_buffer")
    from queue import Queue

    assert isinstance(instance.message_buffer, Queue)
    assert instance.message_buffer.maxsize == 64

    assert instance.descriptor_for_LLM == "Face Presence Sensor"


@pytest.fixture
def face_presence_instance(mock_io_provider):
    config = FacePresenceConfig()
    mock_provider_instance = Mock()
    with (
        patch(
            "inputs.plugins.face_presence_input.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.face_presence_input.FacePresenceProvider",
            return_value=mock_provider_instance,
        ),
    ):
        instance = FacePresence(config=config)
    return instance


@pytest.mark.asyncio
async def test_poll_returns_message_from_buffer(face_presence_instance):
    test_message = "present=[alice], unknown=0, ts=123456"
    face_presence_instance.message_buffer.put_nowait(test_message)

    result = await face_presence_instance._poll()

    assert result == test_message


@pytest.mark.asyncio
async def test_poll_returns_none_if_buffer_empty(face_presence_instance):
    result = await face_presence_instance._poll()

    assert result is None


@pytest.mark.asyncio
async def test_poll_has_delay(face_presence_instance):
    with patch("asyncio.sleep") as mock_sleep:
        await face_presence_instance._poll()
        mock_sleep.assert_called_once_with(0.5)


def test_handle_face_message_adds_to_buffer_successfully(face_presence_instance):
    test_message = "present=[bob], unknown=1, ts=123457"
    initial_size = face_presence_instance.message_buffer.qsize()

    face_presence_instance._handle_face_message(test_message)

    final_size = face_presence_instance.message_buffer.qsize()
    assert final_size == initial_size + 1
    assert face_presence_instance.message_buffer.get_nowait() == test_message


def test_handle_face_message_drops_oldest_on_full_buffer(face_presence_instance):
    for i in range(64):
        face_presence_instance.message_buffer.put_nowait(f"msg_{i}")

    face_presence_instance._handle_face_message("msg_NEW")

    assert face_presence_instance.message_buffer.qsize() == 64

    first_popped = face_presence_instance.message_buffer.get_nowait()
    assert first_popped == "msg_1"

    remaining_items = []
    for _ in range(62):
        remaining_items.append(face_presence_instance.message_buffer.get_nowait())

    last_item = face_presence_instance.message_buffer.get_nowait()
    assert last_item == "msg_NEW"

    assert "msg_0" not in remaining_items


@pytest.mark.asyncio
async def test_raw_to_text_converts_string_to_message(face_presence_instance):
    test_data_str = "present=[charlie], unknown=2, ts=123458"
    timestamp_before = time.time()

    result = await face_presence_instance._raw_to_text(test_data_str)

    timestamp_after = time.time()
    assert result is not None
    assert result.message == test_data_str
    assert timestamp_before <= result.timestamp <= timestamp_after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_if_input_none(face_presence_instance):
    result = await face_presence_instance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_message_to_buffer(face_presence_instance):
    test_data_str = "present=[diana], unknown=0, ts=123459"
    initial_len = len(face_presence_instance.messages)

    with patch("time.time", return_value=1234.0):
        await face_presence_instance.raw_to_text(test_data_str)

    assert len(face_presence_instance.messages) == initial_len + 1
    assert face_presence_instance.messages[-1].message == test_data_str
    assert face_presence_instance.messages[-1].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_does_nothing_if_input_none(face_presence_instance):
    initial_len = len(face_presence_instance.messages)
    await face_presence_instance.raw_to_text(None)

    assert len(face_presence_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(face_presence_instance):
    result = face_presence_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    face_presence_instance, mock_io_provider
):
    from inputs.plugins.face_presence_input import Message

    msg = Message(timestamp=1234.0, message="present=[eve], unknown=1, ts=123460")
    face_presence_instance.messages = [msg]

    result = face_presence_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "Face Presence Sensor" in result
    assert "present=[eve], unknown=1, ts=123460" in result
    assert len(face_presence_instance.messages) == 0
    mock_io_provider.add_input.assert_called_once_with(
        "FacePresence", "present=[eve], unknown=1, ts=123460", 1234.0
    )
