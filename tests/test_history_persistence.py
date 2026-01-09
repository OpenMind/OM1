"""
Tests for conversation history persistence.
"""

import json
import os
import tempfile

import pytest

from providers.history_persistence import clear_history, load_history, save_history


class TestSaveHistory:
    """Tests for save_history function."""

    def test_save_history_creates_file(self):
        """Test that save_history creates the history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]

            result = save_history(history, filepath)

            assert result is True
            assert os.path.exists(filepath)

    def test_save_history_writes_correct_content(self):
        """Test that save_history writes correct JSON content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            history = [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"},
            ]

            save_history(history, filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded == history

    def test_save_history_creates_directory(self):
        """Test that save_history creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "deep", "history.json")
            history = [{"role": "user", "content": "Hello"}]

            result = save_history(history, filepath)

            assert result is True
            assert os.path.exists(filepath)

    def test_save_history_overwrites_existing(self):
        """Test that save_history overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            old_history = [{"role": "user", "content": "Old message"}]
            new_history = [{"role": "user", "content": "New message"}]

            save_history(old_history, filepath)
            save_history(new_history, filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded == new_history

    def test_save_history_handles_unicode(self):
        """Test that save_history handles unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            history = [
                {"role": "user", "content": "Merhaba dünya! 你好世界"},
            ]

            result = save_history(history, filepath)

            assert result is True
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded[0]["content"] == "Merhaba dünya! 你好世界"

    def test_save_empty_history(self):
        """Test that save_history handles empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")

            result = save_history([], filepath)

            assert result is True
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == []


class TestLoadHistory:
    """Tests for load_history function."""

    def test_load_history_reads_file(self):
        """Test that load_history reads existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history, f)

            loaded = load_history(filepath)

            assert loaded == history

    def test_load_history_missing_file(self):
        """Test that load_history returns empty list for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nonexistent.json")

            loaded = load_history(filepath)

            assert loaded == []

    def test_load_history_invalid_json(self):
        """Test that load_history handles corrupted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("{ invalid json }")

            loaded = load_history(filepath)

            assert loaded == []

    def test_load_history_validates_structure(self):
        """Test that load_history validates message structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            invalid_data = [
                {"role": "user", "content": "Valid message"},
                {"invalid": "message"},
                {"role": "assistant"},
                {"content": "No role"},
                {"role": "user", "content": "Another valid"},
            ]

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(invalid_data, f)

            loaded = load_history(filepath)

            assert len(loaded) == 2
            assert loaded[0]["content"] == "Valid message"
            assert loaded[1]["content"] == "Another valid"

    def test_load_history_non_list_data(self):
        """Test that load_history handles non-list data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"not": "a list"}, f)

            loaded = load_history(filepath)

            assert loaded == []


class TestClearHistory:
    """Tests for clear_history function."""

    def test_clear_history_deletes_file(self):
        """Test that clear_history deletes the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([{"role": "user", "content": "test"}], f)

            result = clear_history(filepath)

            assert result is True
            assert not os.path.exists(filepath)

    def test_clear_history_missing_file(self):
        """Test that clear_history succeeds for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nonexistent.json")

            result = clear_history(filepath)

            assert result is True


class TestRoundTrip:
    """Tests for save/load roundtrip."""

    def test_save_load_roundtrip(self):
        """Test that saved history can be loaded back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")
            original = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
                {"role": "assistant", "content": "Second response"},
            ]

            save_history(original, filepath)
            loaded = load_history(filepath)

            assert loaded == original

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "history.json")

            for i in range(5):
                history = [
                    {"role": "user", "content": f"Message {i}"},
                    {"role": "assistant", "content": f"Response {i}"},
                ]
                save_history(history, filepath)
                loaded = load_history(filepath)
                assert loaded == history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
