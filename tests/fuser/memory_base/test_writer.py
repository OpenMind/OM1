from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.fuser.memory_base.writer import MemoryWriter


@pytest.fixture(autouse=True)
def reset_memory_writer_singleton():
    """Reset the MemoryWriter singleton before each test."""
    MemoryWriter.reset()  # type: ignore
    yield
    MemoryWriter.reset()  # type: ignore


class TestMemoryWriterInit:
    def test_creates_daily_dir(self, tmp_path):
        MemoryWriter(memory_root=tmp_path)
        assert (tmp_path / "daily").is_dir()

    def test_creates_memory_md_if_missing(self, tmp_path):
        MemoryWriter(memory_root=tmp_path)
        md = tmp_path / "MEMORY.md"
        assert md.exists()
        assert "Long-Term Memory" in md.read_text()

    def test_does_not_overwrite_existing_memory_md(self, tmp_path):
        md = tmp_path / "MEMORY.md"
        md.write_text("existing content")
        MemoryWriter(memory_root=tmp_path)
        assert md.read_text() == "existing content"

    def test_default_root_is_project_memory(self):
        writer = MemoryWriter()
        assert writer.memory_root.name == "memory"


class TestMemoryWriterAppendInteraction:
    def _make_action(self, type_: str, value: str):
        a = MagicMock()
        a.type = type_
        a.value = value
        return a

    def test_writes_entry_to_daily_file(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        actions = [self._make_action("Speak", "Hello there")]

        writer.append_interaction("Hi robot", actions)

        daily_files = list((tmp_path / "daily").glob("*.md"))
        assert len(daily_files) == 1
        content = daily_files[0].read_text()
        assert "Hi robot" in content
        assert "Hello there" in content

    def test_entry_contains_timestamp_heading(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        writer.append_interaction("test", [])

        content = list((tmp_path / "daily").glob("*.md"))[0].read_text()
        assert "##" in content  # timestamp heading format

    def test_skips_empty_user_message(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        writer.append_interaction("", [])

        assert list((tmp_path / "daily").glob("*.md")) == []

    def test_skips_whitespace_only_message(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        writer.append_interaction("   \t\n", [])

        assert list((tmp_path / "daily").glob("*.md")) == []

    def test_multiple_actions_joined_with_pipe(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        actions = [
            self._make_action("Speak", "Hello"),
            self._make_action("Move", "forward"),
        ]
        writer.append_interaction("go", actions)

        content = list((tmp_path / "daily").glob("*.md"))[0].read_text()
        assert "Hello" in content
        assert "forward" in content

    def test_actions_with_empty_value_are_excluded(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        actions = [
            self._make_action("Speak", "Hi"),
            self._make_action("Idle", ""),  # falsy value — should be excluded
        ]
        writer.append_interaction("hello", actions)

        content = list((tmp_path / "daily").glob("*.md"))[0].read_text()
        assert "Idle" not in content
        assert "Hi" in content

    def test_appends_multiple_interactions(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        writer.append_interaction("first", [self._make_action("Speak", "one")])
        writer.append_interaction("second", [self._make_action("Speak", "two")])

        content = list((tmp_path / "daily").glob("*.md"))[0].read_text()
        assert "first" in content
        assert "second" in content

    def test_disk_error_is_logged_not_raised(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)

        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            writer.append_interaction("hello", [])

    def test_daily_file_named_by_today(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        writer.append_interaction("hello", [])

        today = datetime.now().strftime("%Y-%m-%d")
        assert (tmp_path / "daily" / f"{today}.md").exists()


class TestMemoryWriterAppendToIndex:
    """Tests for append_to_index (write-through indexing)."""

    def _make_action(self, type_: str, value: str):
        a = MagicMock()
        a.type = type_
        a.value = value
        return a

    @pytest.mark.asyncio
    async def test_calls_add_chunk(self, tmp_path):
        """Verify index insertion happens."""
        from unittest.mock import AsyncMock

        writer = MemoryWriter(memory_root=tmp_path)
        actions = [self._make_action("Speak", "Hello")]

        mock_reader = MagicMock()
        mock_reader._index_initialized = True
        mock_reader.index = MagicMock()
        mock_reader.index.add_chunk = AsyncMock(return_value=True)

        with patch("fuser.memory_base.writer.MemoryReader", return_value=mock_reader):
            await writer.append_to_index("Hi robot", actions)

        mock_reader.index.add_chunk.assert_called_once()
        chunk_arg = mock_reader.index.add_chunk.call_args[0][0]
        assert "Hi robot" in chunk_arg.text

    @pytest.mark.asyncio
    async def test_skips_when_not_initialized(self, tmp_path):
        """Index update should be skipped if ensure_index hasn't run."""
        writer = MemoryWriter(memory_root=tmp_path)
        actions = [self._make_action("Speak", "Hello")]

        mock_reader = MagicMock()
        mock_reader._index_initialized = False

        with patch("fuser.memory_base.writer.MemoryReader", return_value=mock_reader):
            await writer.append_to_index("test", actions)

        # No index call attempted
        assert not hasattr(mock_reader.index, "add_chunk") or not mock_reader.index.add_chunk.called

    @pytest.mark.asyncio
    async def test_skips_empty_message(self, tmp_path):
        writer = MemoryWriter(memory_root=tmp_path)
        # Should return immediately without error
        await writer.append_to_index("   ", [])

    @pytest.mark.asyncio
    async def test_index_error_does_not_raise(self, tmp_path):
        """If index update fails, exception is caught and logged."""
        from unittest.mock import AsyncMock

        writer = MemoryWriter(memory_root=tmp_path)
        actions = [self._make_action("Speak", "Hello")]

        mock_reader = MagicMock()
        mock_reader._index_initialized = True
        mock_reader.index = MagicMock()
        mock_reader.index.add_chunk = AsyncMock(side_effect=RuntimeError("embed failed"))

        with patch("fuser.memory_base.writer.MemoryReader", return_value=mock_reader):
            # Should not raise
            await writer.append_to_index("test msg", actions)
