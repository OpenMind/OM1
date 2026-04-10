import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.fuser.memory_base.summarizer import MemorySummarizer


@pytest.fixture(autouse=True)
def reset_summarizer_singleton():
    """Reset the MemorySummarizer singleton before each test."""
    MemorySummarizer.reset()  # type: ignore
    yield
    MemorySummarizer.reset()  # type: ignore


def _make_client(*responses: str):
    """Create a mock client returning responses in sequence."""
    client = MagicMock()
    resp_iter = iter(responses)

    async def _create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = next(resp_iter, "")
        return resp

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def _setup(tmp_path: Path, memory: str = "", daily: dict = None) -> Path:
    """Set up memory directory structure."""
    (tmp_path / "daily").mkdir(parents=True, exist_ok=True)
    mem = tmp_path / "MEMORY.md"
    mem.write_text(memory or "# Long-Term Memory\n")
    if daily:
        for name, content in daily.items():
            (tmp_path / "daily" / name).write_text(content)
    return tmp_path


class TestReadLastSummary:
    def test_no_marker_returns_none(self, tmp_path):
        root = _setup(tmp_path)
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        assert s._read_last_summary() is None

    def test_parses_marker(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2026-04-08 14:30 -->\n# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        assert s._read_last_summary() == datetime(2026, 4, 8, 14, 30)

    def test_missing_file_returns_none(self, tmp_path):
        s = MemorySummarizer(memory_root=tmp_path, client=MagicMock())
        assert s._read_last_summary() is None


class TestWriteLastSummary:
    def test_inserts_marker_when_absent(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._write_last_summary()
        assert "<!-- last_summary:" in s.memory_file.read_text()

    def test_updates_existing_marker(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2026-01-01 00:00 -->\n# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._write_last_summary()
        content = s.memory_file.read_text()
        assert "2026-01-01" not in content
        assert "<!-- last_summary:" in content


class TestFindUnprocessed:
    def test_all_files_when_no_marker(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-07.md": "a", "2026-04-08.md": "b"})
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        assert len(s._find_unprocessed(None)) == 2

    def test_filters_by_date(self, tmp_path):
        root = _setup(
            tmp_path,
            daily={"2026-04-06.md": "old", "2026-04-07.md": "boundary", "2026-04-08.md": "new"},
        )
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        names = [f.stem for f in s._find_unprocessed(datetime(2026, 4, 7))]
        assert "2026-04-06" not in names
        assert "2026-04-07" in names
        assert "2026-04-08" in names

    def test_empty_dir(self, tmp_path):
        root = _setup(tmp_path)
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        assert s._find_unprocessed(None) == []

    def test_skips_non_date_files(self, tmp_path):
        root = _setup(tmp_path, daily={"notes.md": "x", "2026-04-08.md": "y"})
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        assert len(s._find_unprocessed(None)) == 1


class TestExtractCandidates:
    @pytest.mark.asyncio
    async def test_returns_candidates(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("- [IDENTITY] User name is Alice")
        s = MemorySummarizer(memory_root=root, client=client)
        result = await s._extract_candidates("some log")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_none(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("NONE")
        s = MemorySummarizer(memory_root=root, client=client)
        assert await s._extract_candidates("trivial log") == ""


class TestScoreCandidates:
    @pytest.mark.asyncio
    async def test_parses_json_response(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        decisions = [
            {"fact": "User is Alice", "durability": 5, "novelty": 5, "significance": 4, "decision": "PROMOTE"},
        ]
        client = _make_client(json.dumps(decisions))
        s = MemorySummarizer(memory_root=root, client=client)
        result = await s._score_candidates("- [IDENTITY] User is Alice")
        assert len(result) == 1
        assert result[0]["decision"] == "PROMOTE"

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self, tmp_path):
        root = _setup(tmp_path)
        decisions = [{"fact": "x", "decision": "SKIP"}]
        client = _make_client(f"```json\n{json.dumps(decisions)}\n```")
        s = MemorySummarizer(memory_root=root, client=client)
        result = await s._score_candidates("- x")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("not valid json at all")
        s = MemorySummarizer(memory_root=root, client=client)
        assert await s._score_candidates("- fact") == []


class TestApplyDecisions:
    def test_promote_appends(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._apply_decisions(
            [
                {"fact": "User is Alice", "decision": "PROMOTE"},
            ]
        )
        content = s.memory_file.read_text()
        assert "User is Alice" in content
        assert "Dreaming" in content

    def test_update_replaces(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n- User lives in Beijing\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._apply_decisions(
            [
                {"fact": "User lives in Shanghai", "decision": "UPDATE", "replaces": "User lives in Beijing"},
            ]
        )
        content = s.memory_file.read_text()
        assert "Shanghai" in content
        assert "Beijing" not in content

    def test_update_without_match_promotes(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._apply_decisions(
            [
                {"fact": "new fact", "decision": "UPDATE", "replaces": "nonexistent"},
            ]
        )
        assert "new fact" in s.memory_file.read_text()

    def test_skip_does_nothing(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._apply_decisions(
            [
                {"fact": "trivial", "decision": "SKIP"},
            ]
        )
        assert "trivial" not in s.memory_file.read_text()

    def test_empty_decisions_noop(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        original = s.memory_file.read_text()
        s._apply_decisions([])
        assert s.memory_file.read_text() == original


class TestDreamingPipeline:
    @pytest.mark.asyncio
    async def test_full_promote_flow(self, tmp_path):
        """End-to-end: daily log → extract → score → promote."""
        root = _setup(
            tmp_path,
            memory="# Memory\n",
            daily={"2026-04-09.md": "## 10:00\n- **User**: my name is Alice\n"},
        )
        decisions = [
            {"fact": "User's name is Alice", "durability": 5, "novelty": 5, "significance": 5, "decision": "PROMOTE"},
        ]
        client = _make_client(
            "- [IDENTITY] User's name is Alice",  # Stage 1
            json.dumps(decisions),  # Stage 2
        )
        s = MemorySummarizer(memory_root=root, client=client)
        await s.run()

        content = s.memory_file.read_text()
        assert "Alice" in content
        assert "<!-- last_summary:" in content

    @pytest.mark.asyncio
    async def test_skips_when_no_new_files(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2099-12-31 23:59 -->\n# Memory\n")
        client = _make_client()
        s = MemorySummarizer(memory_root=root, client=client)
        await s.run()
        # No LLM calls needed
        assert client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_guard(self, tmp_path):
        root = _setup(tmp_path)
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._running = True
        await s.run()
        assert s._running is True  # unchanged

    @pytest.mark.asyncio
    async def test_running_reset_on_error(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-09.md": "data"})
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        s = MemorySummarizer(memory_root=root, client=client)
        await s.run()
        assert s._running is False

    @pytest.mark.asyncio
    async def test_none_candidates_skips_scoring(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-09.md": "trivial"})
        client = _make_client("NONE")
        s = MemorySummarizer(memory_root=root, client=client)
        await s.run()
        # Only 1 LLM call (extract), no score call
        assert client.chat.completions.create.call_count == 1


class TestSafeWrite:
    def test_atomic_write(self, tmp_path):
        root = _setup(tmp_path, "original")
        s = MemorySummarizer(memory_root=root, client=MagicMock())
        s._safe_write("new content")
        assert s.memory_file.read_text() == "new content"
        assert not s.memory_file.with_suffix(".tmp").exists()
