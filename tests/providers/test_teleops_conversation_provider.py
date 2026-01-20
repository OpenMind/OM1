import time
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from providers.teleops_conversation_provider import (
    ConversationMessage,
    MessageType,
    TeleopsConversationProvider,
)


def reset_teleops_provider() -> None:
    """Reset singleton instance between tests."""
    TeleopsConversationProvider.reset()  # type: ignore[attr-defined]


@pytest.fixture
def provider() -> Generator[TeleopsConversationProvider, None, None]:
    reset_teleops_provider()
    instance = TeleopsConversationProvider(api_key="test-key", base_url="https://example.com")
    yield instance
    reset_teleops_provider()


def test_conversation_message_roundtrip() -> None:
    ts = time.time()
    original = ConversationMessage(
        message_type=MessageType.USER,
        content="hello",
        timestamp=ts,
    )

    as_dict = original.to_dict()
    restored = ConversationMessage.from_dict(as_dict)

    assert restored.message_type == MessageType.USER
    assert restored.content == "hello"
    assert restored.timestamp == ts


def test_is_enabled_true_when_api_key_present() -> None:
    reset_teleops_provider()
    provider = TeleopsConversationProvider(api_key="token")
    assert provider.is_enabled()


def test_is_enabled_false_when_api_key_missing_or_empty() -> None:
    reset_teleops_provider()
    provider_none = TeleopsConversationProvider(api_key=None)
    assert not provider_none.is_enabled()

    reset_teleops_provider()
    provider_empty = TeleopsConversationProvider(api_key="")
    assert not provider_empty.is_enabled()


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_message_worker_skips_when_api_key_missing(mock_post: MagicMock) -> None:
    reset_teleops_provider()
    provider = TeleopsConversationProvider(api_key=None)

    msg = ConversationMessage(
        message_type=MessageType.USER,
        content="hello",
        timestamp=time.time(),
    )

    provider._store_message_worker(msg)  # type: ignore[attr-defined]
    mock_post.assert_not_called()


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_message_worker_skips_empty_content(mock_post: MagicMock, provider: TeleopsConversationProvider) -> None:
    msg = ConversationMessage(
        message_type=MessageType.USER,
        content="   ",  # whitespace only
        timestamp=time.time(),
    )

    provider._store_message_worker(msg)  # type: ignore[attr-defined]
    mock_post.assert_not_called()


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_message_worker_sends_request_on_valid_message(
    mock_post: MagicMock, provider: TeleopsConversationProvider
) -> None:
    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    mock_post.return_value = response

    msg = ConversationMessage(
        message_type=MessageType.USER,
        content="hello world",
        timestamp=time.time(),
    )

    provider._store_message_worker(msg)  # type: ignore[attr-defined]

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == provider.base_url
    assert kwargs["headers"]["Authorization"] == f"Bearer {provider.api_key}"
    assert kwargs["json"] == msg.to_dict()
    assert kwargs["timeout"] == 2


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_message_worker_handles_non_200_response(
    mock_post: MagicMock, provider: TeleopsConversationProvider
) -> None:
    response = MagicMock()
    response.status_code = 500
    response.text = "error"
    mock_post.return_value = response

    msg = ConversationMessage(
        message_type=MessageType.ROBOT,
        content="status update",
        timestamp=time.time(),
    )

    # Should not raise even on non-200
    provider._store_message_worker(msg)  # type: ignore[attr-defined]
    mock_post.assert_called_once()


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_message_worker_handles_exception(mock_post: MagicMock, provider: TeleopsConversationProvider) -> None:
    mock_post.side_effect = Exception("network error")

    msg = ConversationMessage(
        message_type=MessageType.USER,
        content="hello",
        timestamp=time.time(),
    )

    # Should swallow exceptions and not propagate
    provider._store_message_worker(msg)  # type: ignore[attr-defined]


def test_store_message_submits_to_executor(provider: TeleopsConversationProvider) -> None:
    message = ConversationMessage(
        message_type=MessageType.USER,
        content="queued message",
        timestamp=time.time(),
    )

    provider.executor = MagicMock()

    provider._store_message(message)  # type: ignore[attr-defined]

    provider.executor.submit.assert_called_once()
    submit_args, submit_kwargs = provider.executor.submit.call_args
    assert submit_args[0] == provider._store_message_worker  # type: ignore[attr-defined]
    assert isinstance(submit_args[1], ConversationMessage)

