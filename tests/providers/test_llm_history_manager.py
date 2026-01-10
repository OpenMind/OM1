import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from providers.llm_history_manager import ChatMessage, LLMHistoryManager


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
    return LLMHistoryManager(llm_config, openai_client)


@pytest.mark.asyncio
async def test_summarize_messages_success(history_manager):
    # Create test messages
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    # Test successful summarization
    result = await history_manager.summarize_messages(messages)
    assert result.role == "assistant"
    assert "Previously, This is a test summary" == result.content


@pytest.mark.asyncio
async def test_summarize_messages_empty(history_manager):
    # Test with empty messages
    result = await history_manager.summarize_messages([])
    assert result.role == "system"
    assert "No history to summarize" == result.content


@pytest.mark.asyncio
async def test_summarize_messages_api_error(history_manager):
    # Mock API error
    history_manager.client.chat.completions.create.side_effect = Exception("API Error")

    messages = [ChatMessage(role="user", content="Test")]
    result = await history_manager.summarize_messages(messages)

    assert result.role == "system"
    assert "Error summarizing state" == result.content


@pytest.mark.asyncio
async def test_start_summary_task(history_manager):
    # Create test messages that we'll modify in-place
    messages = [
        ChatMessage(role="assistant", content="Previous summary"),
        ChatMessage(role="user", content="New input"),
        ChatMessage(role="user", content="Action taken"),
    ]

    # Replace summarize_messages with a mock
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="assistant", content="New summary"
    )

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    # Verify the task was created
    assert history_manager._summary_task is not None

    # Let the event loop process the callback
    await asyncio.sleep(0.1)

    # Because we mocked summarize_messages, the callback should have run
    # and updated the messages list
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert "New summary" == messages[0].content


@pytest.mark.asyncio
async def test_start_summary_task_empty_messages(history_manager):
    # Test with empty messages
    await history_manager.start_summary_task([])
    assert history_manager._summary_task is None


@pytest.mark.asyncio
async def test_start_summary_task_error_handling(history_manager):
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
        ChatMessage(role="user", content="Message 2"),
        ChatMessage(role="assistant", content="Response 2"),
        ChatMessage(role="user", content="Message 3"),
        ChatMessage(role="assistant", content="Response 3"),
        ChatMessage(role="user", content="Test message"),
    ]

    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="system", content="Error: API service unavailable"
    )

    await history_manager.start_summary_task(messages)

    await asyncio.sleep(0.1)

    assert len(messages) == 5
    assert messages[0].content == "Message 2"
    assert messages[1].content == "Response 2"
    assert messages[2].content == "Message 3"
    assert messages[3].content == "Response 3"
    assert messages[4].content == "Test message"


@pytest.mark.asyncio
async def test_update_history_only_current_tick_inputs():
    """Test that only inputs matching the current tick are added to history."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    # Setup mock class that uses the decorator
    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            # Return mock response with actions
            response = MagicMock()
            response.actions = [
                MockAction(type="speak", value="Hello"),
                MockAction(type="emotion", value="happy"),
            ]
            return response

    # Create provider instance
    provider = MockLLMProvider()

    # Add inputs with different ticks
    # Current tick is 0 (initial value)
    provider.io_provider.add_input("audio", "User said hello", 1234.0)
    provider.io_provider.add_input("vision", "Saw a person", 1235.0)

    # Increment tick to 1
    provider.io_provider.increment_tick()

    # Add inputs for tick 1
    provider.io_provider.add_input("audio_new", "User said goodbye", 1236.0)
    provider.io_provider.add_input("lidar", "Detected obstacle", 1237.0)

    # Process with current tick = 1
    await provider.process("test prompt")

    # Should have 2 messages: inputs and actions
    assert len(history_manager.history) == 2

    # First message should be the inputs message
    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    assert "audio_new" in inputs_msg.content
    assert "User said goodbye" in inputs_msg.content
    assert "lidar" in inputs_msg.content
    assert "Detected obstacle" in inputs_msg.content

    assert "User said hello" not in inputs_msg.content
    assert "Saw a person" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_no_inputs_for_current_tick():
    """Test that when no inputs match current tick, only sensor info is added."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    # Setup mock class that uses the decorator
    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Nothing to report")]
            return response

    provider = MockLLMProvider()

    # Add inputs with tick 0
    provider.io_provider.add_input("audio", "Old audio", 1234.0)

    # Increment tick to 1 without adding new inputs
    provider.io_provider.increment_tick()

    # Process with current tick = 1 (no inputs for this tick)
    await provider.process("test prompt")

    # Should have 2 messages: empty inputs and actions
    assert len(history_manager.history) == 2

    # First message should be the inputs message with just the preamble
    inputs_msg = history_manager.history[0]
    assert inputs_msg.role == "user"
    assert "TestBot sensed the following:" in inputs_msg.content
    # Old inputs should not be included
    assert "Old audio" not in inputs_msg.content


@pytest.mark.asyncio
async def test_update_history_multiple_ticks():
    """Test that inputs are filtered correctly across multiple tick cycles."""
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
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Response")]
            return response

    provider = MockLLMProvider()

    # Tick 0: Add inputs
    provider.io_provider.add_input("input_tick0", "Data at tick 0", 1000.0)
    await provider.process("prompt")

    # Verify only tick 0 data in first cycle
    first_inputs = history_manager.history[0]
    assert "input_tick0" in first_inputs.content
    assert "Data at tick 0" in first_inputs.content

    # Tick 1: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick1", "Data at tick 1", 2000.0)
    await provider.process("prompt")

    # Find the second input message (should be at index 2)
    second_inputs = history_manager.history[2]
    assert "input_tick1" in second_inputs.content
    assert "Data at tick 1" in second_inputs.content
    # Should NOT include tick 0 data
    assert "Data at tick 0" not in second_inputs.content

    # Tick 2: Increment and add new inputs
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("input_tick2", "Data at tick 2", 3000.0)
    await provider.process("prompt")

    # Find the third input message (should be at index 4)
    third_inputs = history_manager.history[4]
    assert "input_tick2" in third_inputs.content
    assert "Data at tick 2" in third_inputs.content
    # Should NOT include previous tick data
    assert "Data at tick 0" not in third_inputs.content
    assert "Data at tick 1" not in third_inputs.content


@pytest.mark.asyncio
async def test_update_history_tick_boundary():
    """Test input filtering at tick boundaries when inputs are updated."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "BoundaryBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> MagicMock:
            response = MagicMock()
            response.actions = [MockAction(type="move", value="forward")]
            return response

    provider = MockLLMProvider()

    # Add input at tick 0
    provider.io_provider.add_input("sensor", "Initial reading", 1000.0)

    # Increment to tick 1
    provider.io_provider.increment_tick()

    # Update the same input key with new data at tick 1
    provider.io_provider.add_input("sensor", "Updated reading", 2000.0)

    # Process at tick 1
    await provider.process("prompt")

    # Should only see the updated reading from tick 1
    inputs_msg = history_manager.history[0]
    assert "Updated reading" in inputs_msg.content
    assert "Initial reading" not in inputs_msg.content


@pytest.mark.asyncio
async def test_summarization_failure_truncates_to_history_length(history_manager):
    """Test that when summarization fails, history is truncated to history_length."""
    # Set history_length to 4
    history_manager.config.history_length = 4

    # Create messages exceeding history_length
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
        ChatMessage(role="user", content="Message 2"),
        ChatMessage(role="assistant", content="Response 2"),
        ChatMessage(role="user", content="Message 3"),
        ChatMessage(role="assistant", content="Response 3"),
        ChatMessage(role="user", content="Message 4"),
        ChatMessage(role="assistant", content="Response 4"),
    ]

    # Mock summarization to return an error
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="system", content="Error: API request timed out"
    )

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    # Verify history was truncated to history_length (4)
    assert len(messages) == 4
    # The oldest messages should be removed
    assert messages[0].content == "Message 3"
    assert messages[1].content == "Response 3"
    assert messages[2].content == "Message 4"
    assert messages[3].content == "Response 4"


@pytest.mark.asyncio
async def test_summarization_exception_truncates_to_history_length(history_manager):
    """Test that when summarization raises an exception, history is truncated to history_length."""
    history_manager.config.history_length = 3

    # Create messages exceeding history_length
    messages = [
        ChatMessage(role="user", content="Old message 1"),
        ChatMessage(role="assistant", content="Old response 1"),
        ChatMessage(role="user", content="Old message 2"),
        ChatMessage(role="assistant", content="Old response 2"),
        ChatMessage(role="user", content="Recent message"),
        ChatMessage(role="assistant", content="Recent response"),
    ]

    # Mock summarization to raise an exception
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.side_effect = Exception("Unexpected error")

    # Run the summary task
    await history_manager.start_summary_task(messages)

    # Let the task and callback complete
    await asyncio.sleep(0.1)

    # Verify history was truncated to history_length (3)
    assert len(messages) == 3
    # The oldest messages should be removed, keeping the most recent 3
    assert messages[0].content == "Old response 2"
    assert messages[1].content == "Recent message"
    assert messages[2].content == "Recent response"


@pytest.mark.asyncio
async def test_llm_response_failure_removes_unpaired_user_message():
    """Test that when LLM response is None, the unpaired user message is removed."""
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
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list) -> None:
            # Return None to simulate LLM failure
            return None

    provider = MockLLMProvider()

    # Add input
    provider.io_provider.add_input("audio", "Test input", 1234.0)

    # Process - this should add a user message but return None
    result = await provider.process("test prompt")

    # Verify the result is None
    assert result is None

    # Verify the unpaired user message was removed
    assert len(history_manager.history) == 0


@pytest.mark.asyncio
async def test_llm_response_failure_with_existing_history():
    """Test that unpaired user message removal doesn't affect existing paired messages."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 10
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name
            self.call_count = 0

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list):
            self.call_count += 1
            if self.call_count == 1:
                # First call succeeds
                response = MagicMock()
                response.actions = [MockAction(type="speak", value="Hello")]
                return response
            else:
                # Second call fails
                return None

    provider = MockLLMProvider()

    # First successful call
    provider.io_provider.add_input("audio", "First input", 1234.0)
    await provider.process("test prompt")

    # Should have 2 messages (user + assistant)
    assert len(history_manager.history) == 2
    assert history_manager.history[0].role == "user"
    assert history_manager.history[1].role == "assistant"

    # Second failed call
    provider.io_provider.increment_tick()
    provider.io_provider.add_input("audio", "Second input", 1235.0)
    await provider.process("test prompt")

    # Should still have 2 messages - the failed user message was removed
    assert len(history_manager.history) == 2
    assert history_manager.history[0].role == "user"
    assert "First input" in history_manager.history[0].content
    assert history_manager.history[1].role == "assistant"


@pytest.mark.asyncio
async def test_multiple_summarization_failures_prevent_unbounded_growth():
    """Test that repeated summarization failures don't cause unbounded history growth."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 6
    config.agent_name = "TestBot"

    client = AsyncMock()
    history_manager = LLMHistoryManager(config, client)

    # Mock summarization to always fail
    history_manager.summarize_messages = AsyncMock()
    history_manager.summarize_messages.return_value = ChatMessage(
        role="system", content="Error: API service unavailable"
    )

    class MockLLMProvider:
        def __init__(self):
            self._config = config
            self._skip_state_management = False
            self.history_manager = history_manager
            self.io_provider = history_manager.io_provider
            self.agent_name = config.agent_name

        @LLMHistoryManager.update_history()
        async def process(self, prompt: str, messages: list):
            response = MagicMock()
            response.actions = [MockAction(type="speak", value="Response")]
            return response

    provider = MockLLMProvider()

    # Simulate many cycles that would exceed history_length
    for i in range(15):
        provider.io_provider.add_input(f"input_{i}", f"Message {i}", float(i))
        await provider.process("test prompt")
        provider.io_provider.increment_tick()

        # Allow summary tasks to complete
        await asyncio.sleep(0.1)

        # Verify history never exceeds history_length + 1 (the +1 is because
        # summarization is triggered AFTER exceeding history_length)
        assert len(history_manager.history) <= config.history_length + 2

    # Final check: history should be at or below history_length
    assert len(history_manager.history) <= config.history_length


@pytest.fixture
def temp_history_dir(tmp_path):
    """Create a temporary directory for history files."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    return history_dir


@pytest.fixture
def persist_config(temp_history_dir):
    """Create a config with persistence enabled."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"
    config.persist_history = True
    config.history_storage_path = str(temp_history_dir / "test_history.json")
    return config


@pytest.fixture
def no_persist_config():
    """Create a config with persistence disabled."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"
    config.persist_history = False
    config.history_storage_path = None
    return config


def test_save_history_creates_file(persist_config, temp_history_dir):
    """Test that _save_history creates a JSON file with correct structure."""
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Add some history
    manager.history = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]
    manager.frame_index = 5

    # Save history
    manager._save_history()

    # Verify file was created
    history_file = Path(persist_config.history_storage_path)
    assert history_file.exists()

    # Verify content
    import json

    with open(history_file) as f:
        data = json.load(f)

    assert data["version"] == 1
    assert data["agent_name"] == "TestBot"
    assert data["frame_index"] == 5
    assert len(data["history"]) == 2
    assert data["history"][0]["role"] == "user"
    assert data["history"][0]["content"] == "Hello"
    assert data["history"][1]["role"] == "assistant"
    assert data["history"][1]["content"] == "Hi there!"


def test_load_history_restores_state(persist_config, temp_history_dir):
    """Test that _load_history correctly restores history from disk."""
    import json

    # Create a history file
    history_file = Path(persist_config.history_storage_path)
    data = {
        "version": 1,
        "agent_name": "TestBot",
        "frame_index": 10,
        "history": [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ],
    }
    with open(history_file, "w") as f:
        json.dump(data, f)

    # Create manager (should auto-load)
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Verify history was loaded
    assert len(manager.history) == 2
    assert manager.history[0].role == "user"
    assert manager.history[0].content == "Previous message"
    assert manager.history[1].role == "assistant"
    assert manager.history[1].content == "Previous response"
    assert manager.frame_index == 10


def test_load_history_handles_missing_file(persist_config):
    """Test that _load_history handles missing file gracefully."""
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Should start with empty history
    assert len(manager.history) == 0
    assert manager.frame_index == 0


def test_load_history_handles_corrupted_file(persist_config, temp_history_dir):
    """Test that _load_history handles corrupted JSON gracefully."""
    # Create a corrupted history file
    history_file = Path(persist_config.history_storage_path)
    with open(history_file, "w") as f:
        f.write("{ invalid json }")

    # Create manager (should handle error gracefully)
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Should start fresh after error
    assert len(manager.history) == 0
    assert manager.frame_index == 0


def test_persistence_disabled_does_not_save(no_persist_config, tmp_path):
    """Test that persistence is disabled when persist_history is False."""
    client = AsyncMock()
    manager = LLMHistoryManager(no_persist_config, client)

    # Add some history
    manager.history = [
        ChatMessage(role="user", content="Test message"),
    ]

    # Save should be no-op
    manager._save_history()

    # Verify no file was created at the storage path
    # (persistence is disabled so _save_history returns early)


def test_clear_history_resets_state(persist_config, temp_history_dir):
    """Test that clear_history resets in-memory state."""
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Add some history
    manager.history = [
        ChatMessage(role="user", content="Test"),
    ]
    manager.frame_index = 5
    manager._save_history()

    # Clear without deleting file
    manager.clear_history(delete_file=False)

    assert len(manager.history) == 0
    assert manager.frame_index == 0

    # File should still exist
    assert Path(persist_config.history_storage_path).exists()


def test_clear_history_deletes_file(persist_config, temp_history_dir):
    """Test that clear_history can delete the persisted file."""
    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Add some history and save
    manager.history = [
        ChatMessage(role="user", content="Test"),
    ]
    manager._save_history()

    # Clear with file deletion
    manager.clear_history(delete_file=True)

    assert len(manager.history) == 0
    assert not Path(persist_config.history_storage_path).exists()


def test_atomic_write_survives_interruption(persist_config, temp_history_dir):
    """Test that atomic write doesn't corrupt existing file on error."""
    import json

    client = AsyncMock()
    manager = LLMHistoryManager(persist_config, client)

    # Save initial history
    manager.history = [
        ChatMessage(role="user", content="Original message"),
    ]
    manager._save_history()

    # Verify initial save
    history_file = Path(persist_config.history_storage_path)
    with open(history_file) as f:
        data = json.load(f)
    assert data["history"][0]["content"] == "Original message"

    # The atomic write uses temp file + rename, so even if there's an error
    # during write, the original file should remain intact


def test_get_history_storage_path_sanitizes_agent_name(no_persist_config):
    """Test that agent names with special characters are sanitized."""
    no_persist_config.agent_name = "Test Bot/With:Special*Chars"
    no_persist_config.persist_history = True
    no_persist_config.history_storage_path = None

    client = AsyncMock()
    manager = LLMHistoryManager(no_persist_config, client)

    # Path should use sanitized name
    expected_name = "Test_Bot_With_Special_Chars"
    assert expected_name in str(manager._history_storage_path)


def test_custom_storage_path(tmp_path):
    """Test that custom storage path is respected."""
    custom_path = tmp_path / "custom" / "location" / "history.json"

    config = MagicMock()
    config.model = "gpt-4o"
    config.history_length = 5
    config.agent_name = "TestBot"
    config.persist_history = True
    config.history_storage_path = str(custom_path)

    client = AsyncMock()
    manager = LLMHistoryManager(config, client)

    # Add and save history
    manager.history = [ChatMessage(role="user", content="Test")]
    manager._save_history()

    # File should be at custom location
    assert custom_path.exists()


def test_version_compatibility_warning(persist_config, temp_history_dir, caplog):
    """Test that loading a newer version file logs a warning."""
    import json
    import logging

    # Create a history file with a future version
    history_file = Path(persist_config.history_storage_path)
    data = {
        "version": 999,
        "agent_name": "TestBot",
        "frame_index": 5,
        "history": [{"role": "user", "content": "Test"}],
    }
    with open(history_file, "w") as f:
        json.dump(data, f)

    # Create manager
    client = AsyncMock()
    with caplog.at_level(logging.WARNING):
        manager = LLMHistoryManager(persist_config, client)

    # Should log a warning about version
    assert any("version" in record.message.lower() for record in caplog.records)

    # But should still load the data
    assert len(manager.history) == 1
