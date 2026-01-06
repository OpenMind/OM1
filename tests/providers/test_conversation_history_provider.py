import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from providers.conversation_history_provider import (
    ConversationEntry,
    ConversationHistoryProvider,
)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)
    # Clean up backup files
    for backup in Path(path).parent.glob(f"{Path(path).name}.*"):
        backup.unlink()


@pytest.fixture
def provider(temp_file):
    """Create a fresh provider instance for each test."""
    # Reset singleton
    ConversationHistoryProvider.reset()
    provider = ConversationHistoryProvider(
        filepath=temp_file, enable_auto_save=False, auto_save_interval=5
    )
    yield provider
    provider.clear(delete_file=True)
    ConversationHistoryProvider.reset()


def test_provider_initialization(temp_file):
    """Test that provider initializes correctly."""
    ConversationHistoryProvider.reset()
    provider = ConversationHistoryProvider(filepath=temp_file)
    assert provider._filepath == temp_file
    assert provider._enable_auto_save is True
    assert provider._auto_save_interval == 10
    assert len(provider.get_history()) == 0
    ConversationHistoryProvider.reset()


def test_singleton_behavior(temp_file):
    """Test that provider follows singleton pattern."""
    ConversationHistoryProvider.reset()
    provider1 = ConversationHistoryProvider(filepath=temp_file)
    provider2 = ConversationHistoryProvider()
    assert provider1 is provider2
    ConversationHistoryProvider.reset()


def test_record_prompt(provider):
    """Test recording a prompt."""
    provider.record_prompt(tick=1, prompt="Hello, world!")
    entry = provider.get_entry(1)
    assert entry is not None
    assert entry.tick == 1
    assert entry.prompt == "Hello, world!"
    assert entry.output is None
    assert entry.prompt_timestamp is not None


def test_record_prompt_with_mode(provider):
    """Test recording a prompt with mode tracking."""
    provider.record_prompt(tick=1, prompt="Test prompt", mode="exploration")
    entry = provider.get_entry(1)
    assert entry is not None
    assert entry.mode == "exploration"


def test_record_output(provider):
    """Test recording an output."""
    provider.record_output(tick=1, output="Response text")
    entry = provider.get_entry(1)
    assert entry is not None
    assert entry.tick == 1
    assert entry.output == "Response text"
    assert entry.prompt is None
    assert entry.output_timestamp is not None


def test_record_prompt_then_output(provider):
    """Test recording prompt followed by output for same tick."""
    provider.record_prompt(tick=1, prompt="Question?")
    provider.record_output(tick=1, output="Answer!")
    entry = provider.get_entry(1)
    assert entry.prompt == "Question?"
    assert entry.output == "Answer!"
    assert entry.prompt_timestamp is not None
    assert entry.output_timestamp is not None


def test_save_to_disk(provider, temp_file):
    """Test saving conversation history to disk."""
    provider.record_prompt(tick=1, prompt="Test")
    provider.record_output(tick=1, output="Response")
    result = provider.save_to_disk()
    assert result is True
    assert os.path.exists(temp_file)

    # Verify file content
    with open(temp_file, "r") as f:
        data = json.load(f)
    assert "version" in data
    assert "entries" in data
    assert len(data["entries"]) == 1


def test_load_from_disk(provider, temp_file):
    """Test loading conversation history from disk."""
    provider.record_prompt(tick=1, prompt="Prompt 1")
    provider.record_output(tick=1, output="Output 1")
    provider.record_prompt(tick=2, prompt="Prompt 2")
    provider.save_to_disk()

    # Clear and reload
    provider.clear()
    assert len(provider.get_history()) == 0

    count = provider.load_from_disk()
    assert count == 2
    assert len(provider.get_history()) == 2

    entry1 = provider.get_entry(1)
    assert entry1.prompt == "Prompt 1"
    assert entry1.output == "Output 1"


def test_auto_save(temp_file):
    """Test that auto-save triggers after threshold."""
    ConversationHistoryProvider.reset()
    provider = ConversationHistoryProvider(
        filepath=temp_file, enable_auto_save=True, auto_save_interval=3
    )

    # Add entries below threshold
    provider.record_prompt(tick=1, prompt="P1")
    provider.record_prompt(tick=2, prompt="P2")
    assert not os.path.exists(temp_file)

    # Add entry that crosses threshold
    provider.record_prompt(tick=3, prompt="P3")
    assert os.path.exists(temp_file)

    # Verify saved content
    with open(temp_file, "r") as f:
        data = json.load(f)
    assert len(data["entries"]) == 3

    ConversationHistoryProvider.reset()


def test_corrupted_file_handling(provider, temp_file):
    """Test handling of corrupted JSON files."""
    # Write corrupted JSON
    with open(temp_file, "w") as f:
        f.write("{ invalid json content")

    count = provider.load_from_disk()
    assert count == 0

    # Check that backup was created
    backup_files = list(
        Path(temp_file).parent.glob(f"{Path(temp_file).name}.corrupted.*")
    )
    assert len(backup_files) == 1


def test_clear_with_delete(provider, temp_file):
    """Test clearing history and deleting file."""
    provider.record_prompt(tick=1, prompt="Test")
    provider.save_to_disk()
    assert os.path.exists(temp_file)

    provider.clear(delete_file=True)
    assert len(provider.get_history()) == 0
    assert not os.path.exists(temp_file)


def test_clear_without_delete(provider, temp_file):
    """Test clearing history while keeping file."""
    provider.record_prompt(tick=1, prompt="Test")
    provider.save_to_disk()
    assert os.path.exists(temp_file)

    provider.clear(delete_file=False)
    assert len(provider.get_history()) == 0
    assert os.path.exists(temp_file)


def test_get_recent_entries(provider):
    """Test retrieving recent entries."""
    for i in range(1, 11):
        provider.record_prompt(tick=i, prompt=f"Prompt {i}")

    recent = provider.get_recent_entries(count=5)
    assert len(recent) == 5
    assert recent[0].tick == 10  # Most recent first
    assert recent[4].tick == 6


def test_get_stats(provider):
    """Test getting statistics about history."""
    provider.record_prompt(tick=1, prompt="P1")
    provider.record_output(tick=1, output="O1")
    provider.record_prompt(tick=2, prompt="P2")

    stats = provider.get_stats()
    assert stats["total_entries"] == 2
    assert stats["complete_conversations"] == 1
    assert stats["filepath"] == provider._filepath


def test_force_save(provider, temp_file):
    """Test force saving regardless of auto-save settings."""
    provider.record_prompt(tick=1, prompt="Test")
    assert not os.path.exists(temp_file)

    result = provider.force_save()
    assert result is True
    assert os.path.exists(temp_file)


def test_atomic_write(provider, temp_file):
    """Test that writes are atomic (no corruption on interrupt)."""
    provider.record_prompt(tick=1, prompt="Test")
    provider.save_to_disk()

    # Verify no .tmp files remain
    tmp_files = list(Path(temp_file).parent.glob(f"{Path(temp_file).name}.tmp"))
    assert len(tmp_files) == 0


def test_thread_safety_with_save(provider, temp_file):
    """Test thread safety with concurrent operations."""
    num_threads = 5
    entries_per_thread = 10
    errors = []

    def record_entries(start_tick):
        try:
            for i in range(entries_per_thread):
                tick = start_tick + i
                provider.record_prompt(tick=tick, prompt=f"Prompt {tick}")
                provider.record_output(tick=tick, output=f"Output {tick}")
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(num_threads):
        start_tick = i * entries_per_thread + 1
        thread = threading.Thread(target=record_entries, args=(start_tick,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert len(errors) == 0
    assert len(provider.get_history()) == num_threads * entries_per_thread


def test_empty_save_and_load(provider, temp_file):
    """Test saving and loading empty history."""
    result = provider.save_to_disk()
    assert result is True

    provider.clear()
    count = provider.load_from_disk()
    assert count == 0


def test_conversation_entry_serialization():
    """Test ConversationEntry to_dict and from_dict methods."""
    entry = ConversationEntry(
        tick=1,
        timestamp=1234567890.0,
        mode="exploration",
        prompt="Test prompt",
        output="Test output",
        prompt_timestamp=1234567890.0,
        output_timestamp=1234567891.0,
    )

    # Test to_dict
    data = entry.to_dict()
    assert data["tick"] == 1
    assert data["mode"] == "exploration"
    assert data["prompt"] == "Test prompt"

    # Test from_dict
    entry2 = ConversationEntry.from_dict(data)
    assert entry2.tick == entry.tick
    assert entry2.mode == entry.mode
    assert entry2.prompt == entry.prompt
    assert entry2.output == entry.output


def test_version_mismatch_warning(provider, temp_file, caplog):
    """Test that version mismatches produce warnings."""
    # Create file with different version
    data = {"version": "2.0", "entries": []}
    with open(temp_file, "w") as f:
        json.dump(data, f)

    provider.load_from_disk()
    assert "version mismatch" in caplog.text.lower()


def test_persistence_across_restarts(temp_file):
    """Test that data persists across provider restarts."""
    # First session
    ConversationHistoryProvider.reset()
    provider1 = ConversationHistoryProvider(filepath=temp_file, enable_auto_save=True)
    provider1.record_prompt(tick=1, prompt="Session 1")
    provider1.force_save()
    ConversationHistoryProvider.reset()

    # Second session (simulated restart)
    provider2 = ConversationHistoryProvider(filepath=temp_file, enable_auto_save=True)
    entry = provider2.get_entry(1)
    assert entry is not None
    assert entry.prompt == "Session 1"
    ConversationHistoryProvider.reset()


def test_output_without_prompt(provider):
    """Test recording output before prompt (edge case)."""
    provider.record_output(tick=1, output="Output first")
    entry = provider.get_entry(1)
    assert entry.output == "Output first"
    assert entry.prompt is None

    # Then add prompt
    provider.record_prompt(tick=1, prompt="Prompt second")
    entry = provider.get_entry(1)
    assert entry.prompt == "Prompt second"
    assert entry.output == "Output first"


def test_timestamps(provider):
    """Test that timestamps are recorded correctly."""
    start_time = time.time()
    provider.record_prompt(tick=1, prompt="Test")
    time.sleep(0.1)
    provider.record_output(tick=1, output="Response")

    entry = provider.get_entry(1)
    assert entry.timestamp >= start_time
    assert entry.prompt_timestamp >= start_time
    assert entry.output_timestamp > entry.prompt_timestamp
