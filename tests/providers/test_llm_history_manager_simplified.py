import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from providers.llm_history_manager import ChatMessage
from providers.llm_history_manager_simplified import (
    ACTION_MAP_SIMPLIFIED,
    LLMHistoryManagerSimplified,
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
    client = MagicMock(spec=openai.AsyncClient)

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test summary"

    chat_mock = MagicMock()
    completions_mock = MagicMock()
    completions_mock.create = AsyncMock(return_value=response)
    chat_mock.completions = completions_mock
    client.chat = chat_mock

    return client


@pytest.fixture
def history_manager(llm_config, openai_client):
    return LLMHistoryManagerSimplified(llm_config, openai_client)


@pytest.mark.asyncio
async def test_summarize_messages_adds_prefix(history_manager):
    """Test that summarize_messages adds the 'do not repeat' prefix."""
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    result = await history_manager.summarize_messages(messages)
    assert result.role == "assistant"
    assert result.content == (
        "[Conversation summary - do not repeat] Previously, This is a test summary"
    )


@pytest.mark.asyncio
async def test_summarize_messages_empty(history_manager):
    """Test with empty messages."""
    result = await history_manager.summarize_messages([])
    assert result.role == "system"
    assert "No history to summarize" == result.content


@pytest.mark.asyncio
async def test_summarize_messages_api_error(history_manager):
    """Test that API errors are handled gracefully."""
    history_manager.client.chat.completions.create.side_effect = Exception("API Error")

    messages = [ChatMessage(role="user", content="Test")]
    result = await history_manager.summarize_messages(messages)

    assert result.role == "system"
    assert "Error summarizing state" == result.content


@pytest.mark.asyncio
async def test_start_summary_task(history_manager):
    """Test that summary task runs and updates messages."""
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="assistant", content="New summary"
    )

    await history_manager.start_summary_task(messages)
    await asyncio.sleep(0.1)

    assert history_manager._summary_task is not None
    await asyncio.sleep(0.1)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert "New summary" == messages[0].content


def test_action_map_includes_greeting_conversation():
    """Test that ACTION_MAP_SIMPLIFIED includes greeting_conversation."""
    assert "greeting_conversation" in ACTION_MAP_SIMPLIFIED
    assert "emotion" in ACTION_MAP_SIMPLIFIED
    assert "speak" in ACTION_MAP_SIMPLIFIED
    assert "move" in ACTION_MAP_SIMPLIFIED


def test_action_map_uses_simple_format():
    """Test that ACTION_MAP_SIMPLIFIED uses plain '{}' format without preambles."""
    for key, fmt in ACTION_MAP_SIMPLIFIED.items():
        assert fmt == "{}", f"Expected '{{}}' for {key}, got '{fmt}'"


@pytest.mark.asyncio
async def test_update_history_user_format():
    """Test that inputs are formatted as 'User: ...' instead of sensor-style."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [
                MockAction(type="speak", value="Hello"),
                MockAction(type="emotion", value="happy"),
            ]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "User said hello", 1234.0)
    provider.io_provider.add_input("vision", "Saw a person", 1235.0)

    provider.io_provider.increment_tick()

    provider.io_provider.add_input("audio_new", "User said goodbye", 1236.0)
    provider.io_provider.add_input("lidar", "Detected obstacle", 1237.0)

    await provider.process("test prompt")

    assert len(history_manager.history) == 2

    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    # Simplified format: "User: ..." without input type names
    assert inputs_msg.content.startswith("User: ")
    assert "User said goodbye" in inputs_msg.content
    assert "Detected obstacle" in inputs_msg.content
    # Old tick inputs should not be present
    assert "User said hello" not in inputs_msg.content
    assert "Saw a person" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_no_inputs():
    """Test that when no inputs match current tick, 'User: (no input)' is used."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Nothing to report")]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Old audio", 1234.0)
    provider.io_provider.increment_tick()

    await provider.process("test prompt")

    assert len(history_manager.history) == 2

    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    assert inputs_msg.content == "User: (no input)"
    assert "Old audio" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_multiple_ticks():
    """Test that inputs are filtered correctly across multiple tick cycles."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 10
    config.agent_name = "MultiTickBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Response")]
            return response

    provider = MockLLMProvider()

    # Tick 0: Add inputs
    provider.io_provider.add_input("input_tick0", "Data at tick 0", 1000.0)
    await provider.process("prompt")

    first_inputs = history_manager.history[0]
    assert "Data at tick 0" in first_inputs.content

    # Tick 1: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick1", "Data at tick 1", 2000.0)
    await provider.process("prompt")

    second_inputs = history_manager.history[2]
    assert "Data at tick 1" in second_inputs.content
    assert "Data at tick 0" not in second_inputs.content

    # Tick 2: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick2", "Data at tick 2", 3000.0)
    await provider.process("prompt")

    third_inputs = history_manager.history[4]
    assert "Data at tick 2" in third_inputs.content
    assert "Data at tick 0" not in third_inputs.content
    assert "Data at tick 1" not in third_inputs.content


@pytest.mark.asyncio
async def test_update_history_extracts_json_response():
    """Test that action values with JSON-wrapped response field are extracted."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [
                MockAction(
                    type="greeting_conversation",
                    value='{"response": "Hello there!"}',
                ),
            ]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Hi", 1234.0)
    await provider.process("test prompt")

    assert len(history_manager.history) == 2

    action_msg = history_manager.history[1]
    assert action_msg.role == "assistant"
    # Should extract "Hello there!" from JSON, not show raw JSON
    assert "Hello there!" in action_msg.content
    assert "TestBot:" in action_msg.content


@pytest.mark.asyncio
async def test_update_history_agent_name_format():
    """Test that action messages use '{agent_name}: {text}' format."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "Bits"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Welcome to GTC!")]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Hello", 1234.0)
    await provider.process("test prompt")

    action_msg = history_manager.history[1]
    assert action_msg.content == "Bits: Welcome to GTC!"


@pytest.mark.asyncio
async def test_update_history_llm_failure_removes_unpaired_message():
    """Test that when LLM returns None, unpaired user message is removed."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> None:
            return None

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Test input", 1234.0)
    result = await provider.process("test prompt")

    assert result is None
    assert len(history_manager.history) == 0


@pytest.mark.asyncio
async def test_update_history_skip_when_history_length_zero():
    """Test that history is skipped entirely when history_length is 0."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 0
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManagerSimplified(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManagerSimplified.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            # messages should be empty list when history_length is 0
            assert messages == []
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Hello")]
            return response

    provider = MockLLMProvider()

    provider.io_provider.add_input("audio", "Test input", 1234.0)
    await provider.process("test prompt")

    # History should remain empty
    assert len(history_manager.history) == 0


def test_get_messages_empty(history_manager):
    """Test get_messages returns empty list when no history."""
    result = history_manager.get_messages()
    assert result == []


def test_get_messages_multiple(history_manager):
    """Test get_messages with multiple messages."""
    history_manager.history.extend(
        [
            ChatMessage(role="user", content="User: Hello"),
            ChatMessage(role="assistant", content="Test Robot: Hi there"),
        ]
    )
    result = history_manager.get_messages()
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "User: Hello"}
    assert result[1] == {"role": "assistant", "content": "Test Robot: Hi there"}
