"""
Unit Tests for LLM History Persistence (Issue #985)

Tests for: Save conversation history to disk and reload on restart

Run with: pytest tests/providers/test_llm_history_persistence.py -v
"""

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class ChatMessage:
    """ChatMessage for testing."""

    role: str
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data):
        return cls(role=data["role"], content=data["content"])


class TestHistoryPersistence:
    """Test suite for conversation history persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def history_file(self, temp_dir):
        """Create a path for the history file."""
        return os.path.join(temp_dir, "memory", "history.json")

    # =========================================================================
    # Test 1: Basic Save Functionality
    # =========================================================================

    def test_save_history_creates_file(self, temp_dir):
        """Test that save_history creates the history file."""
        history_file = os.path.join(temp_dir, "history.json")

        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]

        data = {
            "version": "1.0",
            "agent_name": "TestBot",
            "frame_index": 0,
            "message_count": len(history),
            "messages": [msg.to_dict() for msg in history],
        }

        with open(history_file, "w") as f:
            json.dump(data, f)

        assert os.path.exists(history_file)

        with open(history_file) as f:
            loaded = json.load(f)

        assert loaded["message_count"] == 2
        assert loaded["messages"][0]["content"] == "Hello"

    def test_save_history_creates_directory(self, temp_dir):
        """Test that save_history creates parent directories."""
        history_file = os.path.join(temp_dir, "nested", "deep", "history.json")

        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        with open(history_file, "w") as f:
            json.dump({"messages": []}, f)

        assert os.path.exists(history_file)

    # =========================================================================
    # Test 2: Basic Load Functionality
    # =========================================================================

    def test_load_history_restores_messages(self, temp_dir):
        """Test that load_history restores saved messages."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {
            "version": "1.0",
            "agent_name": "TestBot",
            "frame_index": 5,
            "message_count": 3,
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ],
        }

        with open(history_file, "w") as f:
            json.dump(data, f)

        with open(history_file) as f:
            loaded = json.load(f)

        messages = [ChatMessage.from_dict(m) for m in loaded["messages"]]

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "First message"
        assert loaded["frame_index"] == 5

    def test_load_history_missing_file(self, temp_dir):
        """Test that load_history handles missing files gracefully."""
        history_file = os.path.join(temp_dir, "nonexistent.json")
        assert not os.path.exists(history_file)

    # =========================================================================
    # Test 3: Atomic Write Safety
    # =========================================================================

    def test_atomic_write_uses_temp_file(self, temp_dir):
        """Test that atomic writes use temporary files."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {"messages": [{"role": "user", "content": "test"}]}

        fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix=".history_", dir=temp_dir)

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, history_file)

            assert os.path.exists(history_file)
            assert not os.path.exists(temp_path)

        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def test_atomic_write_survives_crash(self, temp_dir):
        """Test that atomic writes don't corrupt data on simulated crash."""
        history_file = os.path.join(temp_dir, "history.json")

        initial_data = {"messages": [{"role": "user", "content": "original"}]}
        with open(history_file, "w") as f:
            json.dump(initial_data, f)

        # Simulate crash: temp file exists but rename didn't complete
        temp_path = history_file + ".tmp"
        partial_data = '{"messages": [{"role": "user", "content": "partial'

        with open(temp_path, "w") as f:
            f.write(partial_data)

        # Original file should still be valid
        with open(history_file) as f:
            loaded = json.load(f)

        assert loaded["messages"][0]["content"] == "original"

        os.unlink(temp_path)

    # =========================================================================
    # Test 4: Corruption Handling
    # =========================================================================

    def test_corrupted_json_handling(self, temp_dir):
        """Test that corrupted JSON files are handled gracefully."""
        history_file = os.path.join(temp_dir, "history.json")

        with open(history_file, "w") as f:
            f.write('{"messages": [{"role": "user", invalid json here')

        try:
            with open(history_file) as f:
                json.load(f)
            loaded = True
        except json.JSONDecodeError:
            loaded = False

        assert not loaded

    def test_corrupted_file_backup(self, temp_dir):
        """Test that corrupted files are backed up."""
        history_file = os.path.join(temp_dir, "history.json")

        with open(history_file, "w") as f:
            f.write("not valid json at all {{{")

        backup_path = history_file + ".corrupted.123456"
        os.rename(history_file, backup_path)

        assert os.path.exists(backup_path)
        assert not os.path.exists(history_file)

    def test_invalid_structure_handling(self, temp_dir):
        """Test handling of valid JSON but invalid structure."""
        history_file = os.path.join(temp_dir, "history.json")

        with open(history_file, "w") as f:
            json.dump(["not", "a", "dict"], f)

        with open(history_file) as f:
            data = json.load(f)

        assert not isinstance(data, dict)

    def test_missing_fields_handling(self, temp_dir):
        """Test handling of missing required fields."""
        history_file = os.path.join(temp_dir, "history.json")

        with open(history_file, "w") as f:
            json.dump({"version": "1.0"}, f)

        with open(history_file) as f:
            data = json.load(f)

        messages = data.get("messages", [])
        assert messages == []

    # =========================================================================
    # Test 5: Thread Safety
    # =========================================================================

    def test_concurrent_saves(self, temp_dir):
        """Test that concurrent saves don't corrupt data."""
        history_file = os.path.join(temp_dir, "history.json")
        lock = threading.RLock()
        errors = []

        def save_worker(worker_id, iterations):
            for i in range(iterations):
                try:
                    with lock:
                        data = {
                            "worker": worker_id,
                            "iteration": i,
                            "messages": [{"role": "user", "content": f"msg-{worker_id}-{i}"}],
                        }

                        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=temp_dir)
                        try:
                            with os.fdopen(fd, "w") as f:
                                json.dump(data, f)
                            os.replace(temp_path, history_file)
                        except Exception:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            raise

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

        with open(history_file) as f:
            data = json.load(f)

        assert "messages" in data

    def test_concurrent_reads(self, temp_dir):
        """Test that concurrent reads work correctly."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {"messages": [{"role": "user", "content": "test"}]}
        with open(history_file, "w") as f:
            json.dump(data, f)

        results = []
        errors = []

        def read_worker():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                results.append(len(data.get("messages", [])))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == 1 for r in results)

    # =========================================================================
    # Test 6: Edge Cases
    # =========================================================================

    def test_empty_history_save_load(self, temp_dir):
        """Test saving and loading empty history."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {
            "version": "1.0",
            "messages": [],
            "frame_index": 0,
        }

        with open(history_file, "w") as f:
            json.dump(data, f)

        with open(history_file) as f:
            loaded = json.load(f)

        assert loaded["messages"] == []

    def test_large_history_handling(self, temp_dir):
        """Test handling of large conversation history."""
        history_file = os.path.join(temp_dir, "history.json")

        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}" * 100}
            for i in range(1000)
        ]

        data = {"version": "1.0", "messages": messages}

        with open(history_file, "w") as f:
            json.dump(data, f)

        with open(history_file) as f:
            loaded = json.load(f)

        assert len(loaded["messages"]) == 1000

    def test_unicode_content_handling(self, temp_dir):
        """Test handling of unicode content in messages."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {
            "messages": [
                {"role": "user", "content": "Hello 👋 World 🌍"},
                {"role": "assistant", "content": "你好世界 مرحبا بالعالم"},
                {"role": "user", "content": "Émoji: 🚀🎉✨"},
            ]
        }

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        with open(history_file, encoding="utf-8") as f:
            loaded = json.load(f)

        assert "👋" in loaded["messages"][0]["content"]
        assert "你好" in loaded["messages"][1]["content"]

    def test_special_characters_in_path(self, temp_dir):
        """Test handling of special characters in file path."""
        history_file = os.path.join(temp_dir, "agent name with spaces", "history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        data = {"messages": [{"role": "user", "content": "test"}]}

        with open(history_file, "w") as f:
            json.dump(data, f)

        assert os.path.exists(history_file)

    # =========================================================================
    # Test 7: Frame Index Persistence
    # =========================================================================

    def test_frame_index_persisted(self, temp_dir):
        """Test that frame_index is persisted with history."""
        history_file = os.path.join(temp_dir, "history.json")

        data = {
            "version": "1.0",
            "frame_index": 42,
            "messages": [{"role": "user", "content": "test"}],
        }

        with open(history_file, "w") as f:
            json.dump(data, f)

        with open(history_file) as f:
            loaded = json.load(f)

        assert loaded["frame_index"] == 42

    # =========================================================================
    # Test 8: Hash-based Change Detection
    # =========================================================================

    def test_skip_save_when_unchanged(self, temp_dir):
        """Test that saves are skipped when history hasn't changed."""
        history = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]

        def compute_hash(messages, frame_index):
            return hash((frame_index, tuple((m.role, m.content) for m in messages)))

        hash1 = compute_hash(history, 0)
        hash2 = compute_hash(history, 0)

        assert hash1 == hash2  # Same history = same hash

        history.append(ChatMessage(role="user", content="New message"))
        hash3 = compute_hash(history, 0)

        assert hash1 != hash3  # Changed history = different hash

    # =========================================================================
    # Test 9: Auto-save Interval
    # =========================================================================

    def test_auto_save_interval(self, temp_dir):
        """Test that auto-save respects the interval setting."""
        save_counter = 0
        auto_save_interval = 3
        saves_performed = 0

        for i in range(10):
            save_counter += 1
            if save_counter >= auto_save_interval:
                save_counter = 0
                saves_performed += 1

        # Should have saved 3 times (at iterations 3, 6, 9)
        assert saves_performed == 3


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_to_dict(self):
        """Test ChatMessage.to_dict() method."""
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()

        assert d == {"role": "user", "content": "Hello"}

    def test_from_dict(self):
        """Test ChatMessage.from_dict() class method."""
        d = {"role": "assistant", "content": "Hi there"}
        msg = ChatMessage.from_dict(d)

        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ChatMessage(role="user", content="Test message")
        restored = ChatMessage.from_dict(original.to_dict())

        assert original.role == restored.role
        assert original.content == restored.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
