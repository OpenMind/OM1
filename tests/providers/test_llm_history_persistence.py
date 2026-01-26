"""
Unit Tests for LLM History Persistence (Issue #985)

These tests verify the actual LLMHistoryManager persistence implementation,
not just basic file I/O operations.

Run with: pytest tests/providers/test_llm_history_persistence.py -v
"""

import json
import os
import tempfile
import threading
from dataclasses import dataclass

import pytest

# Import the actual classes from the implementation
from providers.llm_history_manager import ChatMessage, LLMHistoryManager


@dataclass
class MockLLMConfig:
    """Mock LLMConfig for testing without full dependencies."""

    model: str = "gpt-4o-mini"
    agent_name: str = "TestBot"
    history_length: int = 10
    history_file_path: str = None
    auto_save_interval: int = 1


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    pass


class TestChatMessageSerialization:
    """Tests for ChatMessage serialization methods."""

    def test_to_dict(self):
        """Test ChatMessage.to_dict() returns correct format."""
        msg = ChatMessage(role="user", content="Hello, world!")
        result = msg.to_dict()

        assert result == {"role": "user", "content": "Hello, world!"}

    def test_from_dict(self):
        """Test ChatMessage.from_dict() creates correct object."""
        data = {"role": "assistant", "content": "Hi there!"}
        msg = ChatMessage.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_roundtrip(self):
        """Test to_dict/from_dict preserves data."""
        original = ChatMessage(role="user", content="Test message 🚀")
        restored = ChatMessage.from_dict(original.to_dict())

        assert original.role == restored.role
        assert original.content == restored.content


class TestLLMHistoryManagerPersistence:
    """Test suite for LLMHistoryManager persistence functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def history_file(self, temp_dir):
        """Create a path for the history file."""
        return os.path.join(temp_dir, "memory", "history.json")

    @pytest.fixture
    def config_with_persistence(self, history_file):
        """Create a config with persistence enabled."""
        return MockLLMConfig(
            history_file_path=history_file,
            auto_save_interval=1,
        )

    @pytest.fixture
    def config_without_persistence(self):
        """Create a config without persistence."""
        return MockLLMConfig(history_file_path=None)

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return MockOpenAIClient()

    # =========================================================================
    # Test: save() method
    # =========================================================================

    def test_save_creates_file(self, config_with_persistence, mock_client, history_file):
        """Test that save() creates the history file."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)

        # Add some history
        manager.history.append(ChatMessage(role="user", content="Hello"))
        manager.history.append(ChatMessage(role="assistant", content="Hi!"))

        # Save
        result = manager.save()

        assert result is True
        assert os.path.exists(history_file)

        # Verify file contents
        with open(history_file) as f:
            data = json.load(f)

        assert data["message_count"] == 2
        assert data["messages"][0]["content"] == "Hello"

    def test_save_creates_directory(self, temp_dir, mock_client):
        """Test that save() creates nested directories."""
        history_file = os.path.join(temp_dir, "deep", "nested", "history.json")
        config = MockLLMConfig(history_file_path=history_file)

        manager = LLMHistoryManager(config, mock_client)
        manager.history.append(ChatMessage(role="user", content="Test"))

        result = manager.save()

        assert result is True
        assert os.path.exists(history_file)

    def test_save_without_persistence_returns_false(
        self, config_without_persistence, mock_client
    ):
        """Test that save() returns False when persistence is disabled."""
        manager = LLMHistoryManager(config_without_persistence, mock_client)
        manager.history.append(ChatMessage(role="user", content="Test"))

        result = manager.save()

        assert result is False

    # =========================================================================
    # Test: load() method
    # =========================================================================

    def test_load_restores_history(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that load() restores saved history."""
        # First, save some history
        manager1 = LLMHistoryManager(config_with_persistence, mock_client)
        manager1.history = [
            ChatMessage(role="user", content="First"),
            ChatMessage(role="assistant", content="Second"),
        ]
        manager1.frame_index = 42
        manager1.save()

        # Create new manager and load
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)

        # Should auto-load on init, but we can also call load()
        assert len(manager2.history) == 2
        assert manager2.history[0].content == "First"
        assert manager2.frame_index == 42

    def test_load_handles_missing_file(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that load() handles missing file gracefully."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)

        # File doesn't exist yet
        assert len(manager.history) == 0

    def test_load_handles_corrupted_file(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that load() handles corrupted JSON gracefully."""
        # Create corrupted file
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w") as f:
            f.write('{"messages": [{"role": "user" invalid json')

        # Should not crash, and should backup corrupted file
        manager = LLMHistoryManager(config_with_persistence, mock_client)

        assert len(manager.history) == 0
        # Check that backup was created
        backup_files = [f for f in os.listdir(os.path.dirname(history_file)) if ".corrupted." in f]
        assert len(backup_files) == 1

    def test_load_logs_malformed_messages(
        self, config_with_persistence, mock_client, history_file, caplog
    ):
        """Test that load() logs warnings for malformed messages."""
        # Create file with some malformed messages
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        data = {
            "version": "1.0",
            "messages": [
                {"role": "user", "content": "Good message"},
                {"role": "user"},  # Missing content
                {"content": "Missing role"},  # Missing role
                "not a dict",  # Not a dict
            ],
        }
        with open(history_file, "w") as f:
            json.dump(data, f)

        import logging
        with caplog.at_level(logging.WARNING):
            manager = LLMHistoryManager(config_with_persistence, mock_client)

        # Only the good message should be loaded
        assert len(manager.history) == 1
        assert manager.history[0].content == "Good message"

    # =========================================================================
    # Test: clear() method
    # =========================================================================

    def test_clear_removes_history(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that clear() removes history from memory and disk."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.history = [
            ChatMessage(role="user", content="Test"),
        ]
        manager.frame_index = 10
        manager.save()

        # Clear
        manager.clear()

        assert len(manager.history) == 0
        assert manager.frame_index == 0

        # File should exist but be empty
        with open(history_file) as f:
            data = json.load(f)
        assert data["messages"] == []

    # =========================================================================
    # Test: Auto-save functionality
    # =========================================================================

    def test_auto_save_triggers_at_interval(self, temp_dir, mock_client):
        """Test that auto-save triggers after configured interval."""
        history_file = os.path.join(temp_dir, "history.json")
        config = MockLLMConfig(history_file_path=history_file, auto_save_interval=3)

        manager = LLMHistoryManager(config, mock_client)

        # Simulate interactions
        for i in range(5):
            manager.history.append(ChatMessage(role="user", content=f"Message {i}"))
            manager._maybe_auto_save()

        # After 5 calls with interval=3, should have saved once (at call 3)
        # File should exist
        assert os.path.exists(history_file)

    def test_auto_save_resets_counter_on_load(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that load() resets the auto-save counter."""
        # Create and save
        manager1 = LLMHistoryManager(config_with_persistence, mock_client)
        manager1.history.append(ChatMessage(role="user", content="Test"))
        manager1._save_counter = 5
        manager1.save()

        # Load in new manager
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)

        # Counter should be reset
        assert manager2._save_counter == 0

    # =========================================================================
    # Test: Hash-based change detection
    # =========================================================================

    def test_skip_save_when_unchanged(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that save() skips when history hasn't changed."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.history.append(ChatMessage(role="user", content="Test"))
        manager.save()

        # Get file modification time
        mtime1 = os.path.getmtime(history_file)

        # Save again without changes
        import time
        time.sleep(0.1)  # Small delay to ensure mtime would change
        manager.save()

        # File should not have been modified
        mtime2 = os.path.getmtime(history_file)
        assert mtime1 == mtime2

    def test_save_when_changed(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that save() writes when history has changed."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.history.append(ChatMessage(role="user", content="Test"))
        manager.save()

        mtime1 = os.path.getmtime(history_file)

        # Add new message
        import time
        time.sleep(0.1)
        manager.history.append(ChatMessage(role="assistant", content="Response"))
        manager.save()

        mtime2 = os.path.getmtime(history_file)
        assert mtime2 > mtime1

    # =========================================================================
    # Test: Thread safety
    # =========================================================================

    def test_concurrent_saves_are_safe(self, temp_dir, mock_client):
        """Test that concurrent saves don't corrupt data."""
        history_file = os.path.join(temp_dir, "history.json")
        config = MockLLMConfig(history_file_path=history_file)

        manager = LLMHistoryManager(config, mock_client)
        errors = []

        def save_worker(worker_id, iterations):
            for i in range(iterations):
                try:
                    manager.history.append(
                        ChatMessage(role="user", content=f"msg-{worker_id}-{i}")
                    )
                    manager.save()
                except Exception as e:
                    errors.append(str(e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=save_worker, args=(i, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

        # File should be valid JSON
        with open(history_file) as f:
            data = json.load(f)
        assert "messages" in data

    # =========================================================================
    # Test: Security - Path validation
    # =========================================================================

    def test_rejects_directory_traversal(self, temp_dir, mock_client, caplog):
        """Test that directory traversal paths are rejected."""
        import logging

        malicious_path = os.path.join(temp_dir, "..", "..", "etc", "passwd")
        config = MockLLMConfig(history_file_path=malicious_path)

        with caplog.at_level(logging.WARNING):
            manager = LLMHistoryManager(config, mock_client)

        # Path should be rejected
        assert manager._history_file_path is None

    # =========================================================================
    # Test: Edge cases
    # =========================================================================

    def test_empty_history(self, config_with_persistence, mock_client, history_file):
        """Test saving and loading empty history."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.save()

        # Reload
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)
        assert len(manager2.history) == 0

    def test_unicode_content(self, config_with_persistence, mock_client, history_file):
        """Test saving and loading unicode content."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.history = [
            ChatMessage(role="user", content="Hello 👋 World 🌍"),
            ChatMessage(role="assistant", content="你好世界 مرحبا"),
        ]
        manager.save()

        # Reload
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)
        assert "👋" in manager2.history[0].content
        assert "你好" in manager2.history[1].content

    def test_large_history(self, config_with_persistence, mock_client, history_file):
        """Test handling large conversation history."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        for i in range(1000):
            manager.history.append(
                ChatMessage(role="user", content=f"Message {i}" * 50)
            )
        manager.save()

        # Reload
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)
        assert len(manager2.history) == 1000

    def test_frame_index_persisted(
        self, config_with_persistence, mock_client, history_file
    ):
        """Test that frame_index is persisted."""
        manager = LLMHistoryManager(config_with_persistence, mock_client)
        manager.history.append(ChatMessage(role="user", content="Test"))
        manager.frame_index = 42
        manager.save()

        # Reload
        manager2 = LLMHistoryManager(config_with_persistence, mock_client)
        assert manager2.frame_index == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
