from unittest.mock import patch

import pytest

from providers.teleops_conversation_provider import (
    ConversationMessage,
    MessageType,
    TeleopsConversationProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TeleopsConversationProvider.reset()  # type: ignore
    yield
    TeleopsConversationProvider.reset()  # type: ignore


def test_message_type_enum():
    """Test MessageType enum values."""
    assert MessageType.USER.value == "user"
    assert MessageType.ROBOT.value == "robot"


def test_conversation_message_to_dict():
    """Test converting ConversationMessage to a dictionary."""
    msg = ConversationMessage(
        message_type=MessageType.USER, content="Hello", timestamp=1234567890.0
    )

    msg_dict = msg.to_dict()

    assert msg_dict["type"] == "user"
    assert msg_dict["content"] == "Hello"
    assert msg_dict["timestamp"] == 1234567890.0


def test_conversation_message_from_dict():
    """Test creating ConversationMessage from a dictionary."""
    data = {"type": "robot", "content": "Hi there", "timestamp": 1234567890.0}

    msg = ConversationMessage.from_dict(data)

    assert msg.message_type == MessageType.ROBOT
    assert msg.content == "Hi there"
    assert msg.timestamp == 1234567890.0


def test_initialization_with_api_key():
    """Test initialization with an API key."""
    provider = TeleopsConversationProvider(api_key="test_key")

    assert provider.api_key == "test_key"
    assert provider.base_url == "https://api.openmind.org/api/core/teleops/conversation"


def test_initialization_without_api_key():
    """Test initialization without an API key."""
    provider = TeleopsConversationProvider()

    assert provider.api_key is None


def test_singleton_pattern():
    """Test singleton pattern of the provider."""
    provider1 = TeleopsConversationProvider()
    provider2 = TeleopsConversationProvider()
    assert provider1 is provider2


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_user_message(mock_post):
    """Test storing a user message with an API key."""
    provider = TeleopsConversationProvider(api_key="test_key")

    with patch.object(provider.executor, "submit") as mock_submit:
        provider.store_user_message("Hello")
        assert mock_submit.call_count == 1

        call_args = mock_submit.call_args
        message = call_args[0][1]  # Second argument to submit is the message
        assert message.content == "Hello"
        assert message.message_type == MessageType.USER


@patch("providers.teleops_conversation_provider.requests.post")
def test_store_robot_message(mock_post):
    """Test storing a robot message with an API key."""
    provider = TeleopsConversationProvider(api_key="test_key")

    with patch.object(provider.executor, "submit") as mock_submit:
        provider.store_robot_message("Hi there")
        assert mock_submit.call_count == 1

        call_args = mock_submit.call_args
        message = call_args[0][1]  # Second argument to submit is the message
        assert message.content == "Hi there"
        assert message.message_type == MessageType.ROBOT


def test_store_message_without_api_key():
    """Test storing messages when no API key is provided."""
    provider = TeleopsConversationProvider(api_key=None)

    with patch.object(provider.executor, "submit") as mock_submit:

        provider.store_user_message("Hello")
        provider.store_robot_message("Hi")

        assert mock_submit.call_count == 2

        call_args = mock_submit.call_args
        message = call_args[0][1]  # Second argument to submit is the message
        assert message.content == "Hi"
        assert message.message_type == MessageType.ROBOT

        message = mock_submit.call_args_list[0][0][1]
        assert message.content == "Hello"
        assert message.message_type == MessageType.USER
