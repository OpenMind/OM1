from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.fuser.knowledge_base.base_retriever import Document
from src.fuser.memory_base.reader import (
    MemoryReader,
)


@pytest.fixture(autouse=True)
def reset_memory_reader_singleton():
    """Reset the MemoryReader singleton before each test."""
    MemoryReader.reset()  # type: ignore
    yield
    MemoryReader.reset()  # type: ignore


def _write_memory_md(memory_root: Path, content: str) -> None:
    md = memory_root / "MEMORY.md"
    md.write_text(content, encoding="utf-8")


def _write_daily(daily_dir: Path, filename: str, content: str) -> None:
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / filename).write_text(content, encoding="utf-8")


def _make_doc(text: str, source: str = "2026-04-01.md", score: float = 0.85) -> Document:
    return Document(text=text, metadata={"source": source}, score=score)


class TestReadMemoryMd:
    def test_returns_empty_when_file_missing(self, tmp_path):
        # MemoryReader does not auto-create MEMORY.md — file simply won't exist
        reader = MemoryReader(memory_root=tmp_path)
        assert reader.read_memory_md() == ""

    def test_strips_h1_heading(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        _write_memory_md(tmp_path, "# Long-Term Memory\n\nsome fact\n")
        result = reader.read_memory_md()
        assert "# Long-Term Memory" not in result
        assert "some fact" in result

    def test_strips_html_comments(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        _write_memory_md(tmp_path, "<!-- comment -->\nfact\n")
        result = reader.read_memory_md()
        assert "<!-- comment -->" not in result
        assert "fact" in result

    def test_truncates_to_max_chars(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        _write_memory_md(tmp_path, "x" * 1000)
        result = reader.read_memory_md(max_chars=100)
        assert len(result) <= 103  # 100 chars + "..."
        assert result.endswith("...")

    def test_no_truncation_when_short(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        _write_memory_md(tmp_path, "short content")
        result = reader.read_memory_md()
        assert "..." not in result

    def test_returns_empty_on_read_error(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            result = reader.read_memory_md()
        assert result == ""


class TestFormatContext:
    def _reader(self, tmp_path):
        return MemoryReader(memory_root=tmp_path)

    def test_facts_section_present(self, tmp_path):
        reader = self._reader(tmp_path)
        result = reader.format_context("key fact", [])
        assert "[Facts]" in result
        assert "key fact" in result

    def test_no_facts_no_facts_section(self, tmp_path):
        reader = self._reader(tmp_path)
        result = reader.format_context("", [])
        assert result == ""

    def test_search_results_formatted_with_date(self, tmp_path):
        reader = self._reader(tmp_path)
        doc = _make_doc("[Date: 2026-04-03]\nsome memory", source="2026-04-03.md")
        result = reader.format_context("", [doc])
        assert "[Date: 2026-04-03]" in result
        assert "some memory" in result

    def test_multiple_results_all_included_within_budget(self, tmp_path):
        reader = self._reader(tmp_path)
        docs = [_make_doc(f"memory {i}", source=f"2026-04-0{i+1}.md") for i in range(3)]
        result = reader.format_context("", docs, max_chars=10_000)
        for i in range(3):
            assert f"memory {i}" in result

    def test_respects_max_chars_budget(self, tmp_path):
        reader = self._reader(tmp_path)
        docs = [_make_doc("x" * 500, source="2026-04-01.md") for _ in range(10)]
        result = reader.format_context("", docs, max_chars=200)
        assert len(result) <= 260  # some tolerance for formatting

    def test_sections_separated_by_double_newline(self, tmp_path):
        reader = self._reader(tmp_path)
        doc = _make_doc("day log", source="2026-04-01.md")
        result = reader.format_context("a fact", [doc])
        assert "\n\n" in result


class TestSearchDaily:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        result = await reader.search_daily("")
        assert result == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        result = await reader.search_daily("   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_delegates_to_index_search(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        expected = [_make_doc("hit")]
        reader.index.search = AsyncMock(return_value=expected)
        reader._index_initialized = True  # skip build_index

        result = await reader.search_daily("query", top_k=3)
        assert result == expected
        reader.index.search.assert_called_once_with("query", top_k=3, min_score=reader.min_score)

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        reader._index_initialized = True
        reader.index.search = AsyncMock(side_effect=RuntimeError("oops"))

        result = await reader.search_daily("query")
        assert result == []


class TestEnsureIndex:
    @pytest.mark.asyncio
    async def test_sets_initialized_flag(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)
        assert not reader._index_initialized

        with patch("src.fuser.memory_base.reader.build_index", new_callable=AsyncMock):
            await reader.ensure_index()

        assert reader._index_initialized

    @pytest.mark.asyncio
    async def test_build_index_called_only_once(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)

        with patch("src.fuser.memory_base.reader.build_index", new_callable=AsyncMock) as mock_build:
            await reader.ensure_index()
            await reader.ensure_index()  # second call should be no-op
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_memory_index(self, tmp_path):
        reader = MemoryReader(memory_root=tmp_path)

        with patch("src.fuser.memory_base.reader.build_index", new_callable=AsyncMock):
            index = await reader.ensure_index()

        assert index.__class__.__name__ == "MemoryIndex"
        assert index is reader.index
