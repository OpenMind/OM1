import pytest

from ..context_provider import ContextProvider


def test_conversation_history_add_and_get():
    provider = ContextProvider()

    provider.clear_conversation_history()
    provider.add_message("user", "Hello")
    provider.add_message("assistant", "Hi there")

    history = provider.get_conversation_history()

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there"


def test_conversation_history_clear():
    provider = ContextProvider()

    provider.add_message("user", "Test message")
    provider.clear_conversation_history()

    history = provider.get_conversation_history()
    assert history == []
