import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from inputs.base import Message
from inputs.plugins.gallery_identities_input import (
    GalleryIdentities,
    GalleryIdentitiesConfig,
)


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.gallery_identities_input.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_initialization_creates_providers_and_buffers(mock_io_provider):
    config = GalleryIdentitiesConfig()
    mock_provider_instance = Mock()

    mock_provider_constructor = Mock(return_value=mock_provider_instance)

    with patch(
        "inputs.plugins.gallery_identities_input.GalleryIdentitiesProvider",
        new=mock_provider_constructor,
    ):
        with patch(
            "inputs.plugins.gallery_identities_input.IOProvider",
            return_value=mock_io_provider,
        ):
            instance = GalleryIdentities(config=config)

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

    assert instance.descriptor_for_LLM == "Gallery Identities"


@pytest.fixture
def gallery_identities_instance(mock_io_provider):
    config = GalleryIdentitiesConfig()
    mock_provider_instance = Mock()
    with (
        patch(
            "inputs.plugins.gallery_identities_input.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.gallery_identities_input.GalleryIdentitiesProvider",
            return_value=mock_provider_instance,
        ),
    ):
        instance = GalleryIdentities(config=config)
    return instance


@pytest.mark.asyncio
async def test_poll_returns_message_from_buffer(gallery_identities_instance):
    test_message = "total=2 ids=[alice, bob]"
    gallery_identities_instance.message_buffer.put_nowait(test_message)

    result = await gallery_identities_instance._poll()

    assert result == test_message


@pytest.mark.asyncio
async def test_poll_returns_none_if_buffer_empty(gallery_identities_instance):
    result = await gallery_identities_instance._poll()

    assert result is None


@pytest.mark.asyncio
async def test_poll_has_delay(gallery_identities_instance):
    with patch("asyncio.sleep") as mock_sleep:
        await gallery_identities_instance._poll()
        mock_sleep.assert_called_once_with(0.5)


def test_handle_gallery_message_adds_to_buffer_successfully(
    gallery_identities_instance,
):
    test_message = "total=1 ids=[charlie]"
    initial_size = gallery_identities_instance.message_buffer.qsize()

    gallery_identities_instance._handle_gallery_message(test_message)

    final_size = gallery_identities_instance.message_buffer.qsize()
    assert final_size == initial_size + 1
    assert gallery_identities_instance.message_buffer.get_nowait() == test_message


def test_handle_gallery_message_drops_oldest_on_full_buffer(
    gallery_identities_instance,
):
    for i in range(64):
        gallery_identities_instance.message_buffer.put_nowait(f"id_{i}")

    gallery_identities_instance._handle_gallery_message("id_NEW")

    assert gallery_identities_instance.message_buffer.qsize() == 64

    first_popped = gallery_identities_instance.message_buffer.get_nowait()
    assert first_popped == "id_1"

    remaining_items = []
    for _ in range(62):
        remaining_items.append(gallery_identities_instance.message_buffer.get_nowait())

    last_item = gallery_identities_instance.message_buffer.get_nowait()
    assert last_item == "id_NEW"

    assert "id_0" not in remaining_items


@pytest.mark.asyncio
async def test_raw_to_text_converts_string_to_message(gallery_identities_instance):
    test_data_str = "total=3 ids=[alice, bob, wendy]"
    timestamp_before = time.time()

    result = await gallery_identities_instance._raw_to_text(test_data_str)

    timestamp_after = time.time()
    assert result is not None
    assert result.message == test_data_str
    assert timestamp_before <= result.timestamp <= timestamp_after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_if_input_none(gallery_identities_instance):
    result = await gallery_identities_instance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_message_to_buffer(gallery_identities_instance):
    test_data_str = "total=0 ids=[]"
    initial_len = len(gallery_identities_instance.messages)

    with patch("time.time", return_value=1234.0):
        await gallery_identities_instance.raw_to_text(test_data_str)

    assert len(gallery_identities_instance.messages) == initial_len + 1
    assert gallery_identities_instance.messages[-1].message == test_data_str
    assert gallery_identities_instance.messages[-1].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_does_nothing_if_input_none(gallery_identities_instance):
    initial_len = len(gallery_identities_instance.messages)
    await gallery_identities_instance.raw_to_text(None)

    assert len(gallery_identities_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(gallery_identities_instance):
    result = gallery_identities_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    gallery_identities_instance, mock_io_provider
):
    msg = Message(timestamp=1234.0, message="total=1 ids=[eve]")
    gallery_identities_instance.messages = [msg]

    result = gallery_identities_instance.formatted_latest_buffer()

    assert "INPUT: Gallery Identities" in result
    assert "total=1 ids=[eve]" in result
    assert len(gallery_identities_instance.messages) == 0
    mock_io_provider.add_input.assert_called_once_with(
        "GalleryIdentities", "total=1 ids=[eve]", 1234.0
    )
