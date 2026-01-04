import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from providers.llm_history_manager import (
    ChatMessage,
    LLMHistoryManager,
    MessageRole,
)


@dataclass
class MockAction:
    type: str
    value: str


@pytest.fixture
def llm_config():
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "Test Robot"
    return config


@pytest.fixture
def openai_client():
    client = AsyncMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test summary"
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def history_manager(llm_config, openai_client):
    return LLMHistoryManager(llm_config, openai_client)


@pytest.mark.asyncio
async def test_summarize_messages_success(history_manager):
    messages = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Previous summary"),
        ChatMessage(role=MessageRole.USER, content="New input"),
    ]

    result = await history_manager.summarize_messages(messages)

    assert result is not None
    assert result.role == MessageRole.ASSISTANT
    assert result.content == "Previously, This is a test summary"


@pytest.mark.asyncio
async def test_summarize_messages_empty(history_manager):
    result = await history_manager.summarize_messages([])
    assert result is None


@pytest.mark.asyncio
async def test_summarize_messages_api_error(history_manager):
    history_manager.client.chat.completions.create.side_effect = Exception("API Error")

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    result = await history_manager.summarize_messages(messages)

    assert result is None


@pytest.mark.asyncio
async def test_start_summary_task(history_manager):
    history_manager.history.extend(
        [
            ChatMessage(MessageRole.USER, "Input 1"),
            ChatMessage(MessageRole.ASSISTANT, "Output 1"),
            ChatMessage(MessageRole.USER, "Input 2"),
        ]
    )

    await history_manager.start_summary_task()
    await asyncio.sleep(0.05)

    assert len(history_manager.history) == 1
    assert history_manager.history[0].role == MessageRole.ASSISTANT
    assert "Previously," in history_manager.history[0].content


@pytest.mark.asyncio
async def test_start_summary_task_no_history(history_manager):
    await history_manager.start_summary_task()
    assert history_manager._summary_task is None


@pytest.mark.asyncio
async def test_update_history_only_current_tick_inputs():
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list | None = None):
            response = MagicMock()
            response.actions = [
                MockAction(type="speak", value="Hello"),
                MockAction(type="emotion", value="happy"),
            ]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Old input", 1000.0)
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("vision", "Current input", 2000.0)

    await provider.process("prompt")

    assert len(history_manager.history) == 2

    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == MessageRole.USER
    assert "Current input" in inputs_msg.content
    assert "Old input" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_no_inputs_for_current_tick():
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list | None = None):
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Nothing")]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Old input", 1000.0)
    provider.io_provider.increment_tick()

    await provider.process("prompt")

    assert len(history_manager.history) == 2
    inputs_msg = history_manager.history[0]

    assert "sensed the following" in inputs_msg.content
    assert "Old input" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_multiple_ticks():
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 10
    config.agent_name = "MultiTickBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list | None = None):
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="OK")]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("tick0", "Data 0", 1000.0)
    await provider.process("prompt")

    provider.io_provider.increment_tick()
    provider.io_provider.add_input("tick1", "Data 1", 2000.0)
    await provider.process("prompt")

    provider.io_provider.increment_tick()
    provider.io_provider.add_input("tick2", "Data 2", 3000.0)
    await provider.process("prompt")

    assert "Data 0" in history_manager.history[0].content
    assert "Data 1" in history_manager.history[2].content
    assert "Data 2" in history_manager.history[4].content
