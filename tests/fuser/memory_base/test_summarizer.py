import json
from datetime import datetime
from pathlib import Path
from typing import Optional
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


def _setup(tmp_path: Path, memory: str = "", daily: Optional[dict] = None) -> Path:
    """Set up memory directory structure."""
    (tmp_path / "daily").mkdir(parents=True, exist_ok=True)
    mem = tmp_path / "MEMORY.md"
    mem.write_text(memory or "# Long-Term Memory\n")
    if daily:
        for name, content in daily.items():
            (tmp_path / "daily" / name).write_text(content)
    return tmp_path


def _make_summarizer(root: Path, client: MagicMock | None = None) -> MemorySummarizer:
    """Create a MemorySummarizer with a mocked client for testing."""
    s = MemorySummarizer(memory_root=root, api_key="test-key")
    if client is not None:
        s._client = client
    return s


class TestReadLastSummary:
    def test_no_marker_returns_none(self, tmp_path):
        root = _setup(tmp_path)
        s = _make_summarizer(root)
        assert s._read_last_summary() is None

    def test_parses_marker(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2026-04-08 14:30 -->\n# Memory\n")
        s = _make_summarizer(root)
        assert s._read_last_summary() == datetime(2026, 4, 8, 14, 30)

    def test_missing_file_returns_none(self, tmp_path):
        s = _make_summarizer(tmp_path)
        assert s._read_last_summary() is None


class TestWriteLastSummary:
    def test_inserts_marker_when_absent(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
        s._write_last_summary()
        assert "<!-- last_summary:" in s.memory_file.read_text()

    def test_updates_existing_marker(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2026-01-01 00:00 -->\n# Memory\n")
        s = _make_summarizer(root)
        s._write_last_summary()
        content = s.memory_file.read_text()
        assert "2026-01-01" not in content
        assert "<!-- last_summary:" in content


class TestFindUnprocessed:
    def test_all_files_when_no_marker(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-07.md": "a", "2026-04-08.md": "b"})
        s = _make_summarizer(root)
        assert len(s._find_unprocessed(None)) == 2

    def test_filters_by_date(self, tmp_path):
        root = _setup(
            tmp_path,
            daily={"2026-04-06.md": "old", "2026-04-07.md": "boundary", "2026-04-08.md": "new"},
        )
        s = _make_summarizer(root)
        names = [f.stem for f in s._find_unprocessed(datetime(2026, 4, 7))]
        assert "2026-04-06" not in names
        assert "2026-04-07" in names
        assert "2026-04-08" in names

    def test_same_day_file_not_excluded_by_time(self, tmp_path):
        """File from today should be included even if last_summary has a later time."""
        root = _setup(tmp_path, daily={"2026-04-10.md": "content"})
        s = _make_summarizer(root)
        # last_summary is 14:10 but file date is 00:00 — should still match by date
        result = s._find_unprocessed(datetime(2026, 4, 10, 14, 10))
        assert len(result) == 1

    def test_empty_dir(self, tmp_path):
        root = _setup(tmp_path)
        s = _make_summarizer(root)
        assert s._find_unprocessed(None) == []

    def test_skips_non_date_files(self, tmp_path):
        root = _setup(tmp_path, daily={"notes.md": "x", "2026-04-08.md": "y"})
        s = _make_summarizer(root)
        assert len(s._find_unprocessed(None)) == 1


class TestReadFiles:
    def test_no_filter_without_last_summary(self, tmp_path):
        root = _setup(
            tmp_path,
            daily={
                "2026-04-10.md": "## 10:00:00\n- line1\n\n## 12:00:00\n- line2\n",
            },
        )
        s = _make_summarizer(root)
        files = list((root / "daily").glob("*.md"))
        result = s._read_files(files, last_summary=None)
        assert "line1" in result
        assert "line2" in result

    def test_filters_sections_by_timestamp(self, tmp_path):
        root = _setup(
            tmp_path,
            daily={
                "2026-04-10.md": "## 10:00:00\n- old\n\n## 14:30:00\n- new\n",
            },
        )
        s = _make_summarizer(root)
        files = list((root / "daily").glob("*.md"))
        result = s._read_files(files, last_summary=datetime(2026, 4, 10, 12, 0))
        assert "old" not in result
        assert "new" in result

    def test_boundary_section_excluded(self, tmp_path):
        """Section at exact last_summary time should be excluded (strict >)."""
        root = _setup(
            tmp_path,
            daily={
                "2026-04-10.md": "## 12:00:00\n- exact\n\n## 12:00:01\n- after\n",
            },
        )
        s = _make_summarizer(root)
        files = list((root / "daily").glob("*.md"))
        result = s._read_files(files, last_summary=datetime(2026, 4, 10, 12, 0, 0))
        assert "exact" not in result
        assert "after" in result


class TestCheckEligibility:
    def test_returns_false_when_running(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-10.md": "## 10:00:00\n- a\n" * 20})
        s = _make_summarizer(root)
        s._running = True
        assert s.check_eligibility() is False

    def test_returns_false_when_below_threshold(self, tmp_path):
        root = _setup(
            tmp_path,
            daily={
                "2026-04-10.md": "## 10:00:00\n- a\n\n## 10:01:00\n- b\n",
            },
        )
        s = _make_summarizer(root)
        s.SUMMARY_THRESHOLD = 5
        assert s.check_eligibility() is False

    def test_returns_true_when_above_threshold(self, tmp_path):
        sections = "\n\n".join(f"## 10:{i:02d}:00\n- fact {i}" for i in range(10))
        root = _setup(tmp_path, daily={"2026-04-10.md": sections})
        s = _make_summarizer(root)
        s.SUMMARY_THRESHOLD = 5
        assert s.check_eligibility() is True

    def test_counts_only_new_sections(self, tmp_path):
        sections = "\n\n".join(f"## 14:{i:02d}:00\n- fact {i}" for i in range(5))
        old_sections = "\n\n".join(f"## 10:{i:02d}:00\n- old {i}" for i in range(10))
        root = _setup(
            tmp_path,
            memory="<!-- last_summary: 2026-04-10 13:00 -->\n# Memory\n",
            daily={"2026-04-10.md": old_sections + "\n\n" + sections},
        )
        s = _make_summarizer(root)
        s.SUMMARY_THRESHOLD = 5
        assert s.check_eligibility() is True
        s.SUMMARY_THRESHOLD = 6
        assert s.check_eligibility() is False


class TestExtractCandidates:
    @pytest.mark.asyncio
    async def test_returns_candidates(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("- [IDENTITY] User name is Alice")
        s = _make_summarizer(root, client)
        result = await s._extract_candidates("some log")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_none(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("NONE")
        s = _make_summarizer(root, client)
        assert await s._extract_candidates("trivial log") == ""


class TestScoreCandidates:
    @pytest.mark.asyncio
    async def test_parses_json_response(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        decisions = [
            {
                "fact": "User is Alice",
                "category": "IDENTITY",
                "durability": 5,
                "novelty": 5,
                "significance": 4,
                "decision": "PROMOTE",
            },
        ]
        client = _make_client(json.dumps(decisions))
        s = _make_summarizer(root, client)
        result = await s._score_candidates("- [IDENTITY] User is Alice")
        assert len(result) == 1
        assert result[0]["decision"] == "PROMOTE"
        assert result[0]["category"] == "IDENTITY"

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self, tmp_path):
        root = _setup(tmp_path)
        decisions = [{"fact": "x", "decision": "SKIP"}]
        client = _make_client(f"```json\n{json.dumps(decisions)}\n```")
        s = _make_summarizer(root, client)
        result = await s._score_candidates("- x")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self, tmp_path):
        root = _setup(tmp_path)
        client = _make_client("not valid json at all")
        s = _make_summarizer(root, client)
        assert await s._score_candidates("- fact") == []


class TestApplyDecisions:
    def test_promote_appends_to_category(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "User is Alice", "category": "IDENTITY", "decision": "PROMOTE"},
            ]
        )
        content = s.memory_file.read_text()
        assert "User is Alice" in content
        assert "## Identity" in content

    def test_promote_defaults_to_facts(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "User lives in SF", "decision": "PROMOTE"},
            ]
        )
        content = s.memory_file.read_text()
        assert "## Facts" in content
        assert "User lives in SF" in content

    def test_multiple_categories(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "User is Alice", "category": "IDENTITY", "decision": "PROMOTE"},
                {"fact": "User prefers dark mode", "category": "PREFERENCE", "decision": "PROMOTE"},
                {"fact": "User lives in SF", "category": "FACT", "decision": "PROMOTE"},
            ]
        )
        content = s.memory_file.read_text()
        assert "## Identity" in content
        assert "## Preferences" in content
        assert "## Facts" in content

    def test_appends_to_existing_category(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Identity\n- User is Alice\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "User is 25 years old", "category": "IDENTITY", "decision": "PROMOTE"},
            ]
        )
        content = s.memory_file.read_text()
        assert "User is Alice" in content
        assert "User is 25 years old" in content
        assert content.count("## Identity") == 1

    def test_update_replaces(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n- User lives in Beijing\n")
        s = _make_summarizer(root)
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
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "new fact", "decision": "UPDATE", "replaces": "nonexistent"},
            ]
        )
        assert "new fact" in s.memory_file.read_text()

    def test_skip_does_nothing(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {"fact": "trivial", "decision": "SKIP"},
            ]
        )
        assert "trivial" not in s.memory_file.read_text()

    def test_empty_decisions_noop(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n")
        s = _make_summarizer(root)
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
            daily={"2026-04-09.md": "## 10:00:00\n- **User**: my name is Alice\n"},
        )
        decisions = [
            {
                "fact": "User's name is Alice",
                "category": "IDENTITY",
                "durability": 5,
                "novelty": 5,
                "significance": 5,
                "decision": "PROMOTE",
            },
        ]
        client = _make_client(
            "- [IDENTITY] User's name is Alice",  # Stage 1
            json.dumps(decisions),  # Stage 2
        )
        s = _make_summarizer(root, client)
        await s.run()

        content = s.memory_file.read_text()
        assert "Alice" in content
        assert "## Identity" in content
        assert "<!-- last_summary:" in content

    @pytest.mark.asyncio
    async def test_skips_when_no_new_files(self, tmp_path):
        root = _setup(tmp_path, "<!-- last_summary: 2099-12-31 23:59 -->\n# Memory\n")
        client = _make_client()
        s = _make_summarizer(root, client)
        await s.run()
        assert client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_guard(self, tmp_path):
        root = _setup(tmp_path)
        s = _make_summarizer(root)
        s._running = True
        await s.run()
        assert s._running is True  # unchanged

    @pytest.mark.asyncio
    async def test_running_reset_on_error(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-09.md": "## 10:00:00\n- data\n"})
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        s = _make_summarizer(root, client)
        await s.run()
        assert s._running is False

    @pytest.mark.asyncio
    async def test_none_candidates_skips_scoring(self, tmp_path):
        root = _setup(tmp_path, daily={"2026-04-09.md": "## 10:00:00\n- trivial\n"})
        client = _make_client("NONE")
        s = _make_summarizer(root, client)
        await s.run()
        assert client.chat.completions.create.call_count == 1


class TestSafeWrite:
    def test_atomic_write(self, tmp_path):
        root = _setup(tmp_path, "original")
        s = _make_summarizer(root)
        s._safe_write("new content")
        assert s.memory_file.read_text() == "new content"
        assert not s.memory_file.with_suffix(".tmp").exists()


class TestExtractReviewableFacts:
    """Tests for _extract_reviewable_facts (static, no LLM)."""

    def test_extracts_preferences_and_facts(self):
        content = (
            "# Memory\n\n"
            "## Identity\n- User is Alice\n\n"
            "## Preferences\n- User likes coffee <!-- expired: 0 -->\n\n"
            "## Facts\n- User lives in SF <!-- expired: 2 -->\n"
        )
        result = MemorySummarizer._extract_reviewable_facts(content)
        assert "User likes coffee" in result
        assert "User lives in SF" in result
        assert "User is Alice" not in result  # Identity is skipped

    def test_strips_expired_marker(self):
        content = "## Facts\n- some fact <!-- expired: 3 -->\n"
        result = MemorySummarizer._extract_reviewable_facts(content)
        assert result == ["some fact"]

    def test_handles_no_marker(self):
        content = "## Facts\n- plain fact\n"
        result = MemorySummarizer._extract_reviewable_facts(content)
        assert result == ["plain fact"]

    def test_empty_memory(self):
        assert MemorySummarizer._extract_reviewable_facts("# Memory\n") == []

    def test_no_reviewable_sections(self):
        content = "# Memory\n\n## Identity\n- User is Bob\n"
        assert MemorySummarizer._extract_reviewable_facts(content) == []


class TestApplyExpiration:
    """Tests for _apply_expiration (inline counter logic, no LLM)."""

    def test_increment_expired_count(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- old fact <!-- expired: 0 -->\n")
        s = _make_summarizer(root)
        s._apply_expiration([{"fact": "old fact", "decision": "EXPIRED"}])

        content = s.memory_file.read_text()
        assert "<!-- expired: 1 -->" in content
        assert "old fact" in content

    def test_not_mentioned_fact_unchanged(self, tmp_path):
        """Facts not in LLM output keep their original line."""
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- stable fact <!-- expired: 3 -->\n")
        s = _make_summarizer(root)

        s._apply_expiration([])  # LLM returned no expired facts

        content = s.memory_file.read_text()
        assert "stable fact <!-- expired: 3 -->" in content

    def test_removes_fact_at_threshold(self, tmp_path):
        root = _setup(
            tmp_path,
            "# Memory\n\n## Facts\n- stale fact <!-- expired: 2 -->\n- good fact <!-- expired: 0 -->\n",
        )
        s = _make_summarizer(root)
        s.EXPIRE_THRESHOLD = 3

        s._apply_expiration([{"fact": "stale fact", "decision": "EXPIRED"}])

        content = s.memory_file.read_text()
        assert "stale fact" not in content
        assert "good fact" in content

    def test_empty_decisions_noop(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- fact <!-- expired: 0 -->\n")
        s = _make_summarizer(root)
        original = s.memory_file.read_text()
        s._apply_expiration([])
        assert s.memory_file.read_text() == original

    def test_accumulates_across_calls(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- ephemeral <!-- expired: 0 -->\n")
        s = _make_summarizer(root)
        s.EXPIRE_THRESHOLD = 3

        s._apply_expiration([{"fact": "ephemeral", "decision": "EXPIRED"}])
        assert "<!-- expired: 1 -->" in s.memory_file.read_text()

        s._apply_expiration([{"fact": "ephemeral", "decision": "EXPIRED"}])
        assert "<!-- expired: 2 -->" in s.memory_file.read_text()

        # Third strike — removed
        s._apply_expiration([{"fact": "ephemeral", "decision": "EXPIRED"}])
        assert "ephemeral" not in s.memory_file.read_text()

    def test_fact_without_marker_starts_at_zero(self, tmp_path):
        """Facts without <!-- expired: N --> are treated as count=0."""
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- legacy fact\n")
        s = _make_summarizer(root)

        s._apply_expiration([{"fact": "legacy fact", "decision": "EXPIRED"}])

        content = s.memory_file.read_text()
        assert "<!-- expired: 1 -->" in content
        assert "legacy fact" in content


class TestReviewExpiration:
    """Tests for _review_expiration (LLM call)."""

    @pytest.mark.asyncio
    async def test_calls_llm_with_reviewable_facts(self, tmp_path):
        root = _setup(
            tmp_path,
            "# Memory\n\n## Preferences\n- User likes tea <!-- expired: 0 -->\n\n"
            "## Facts\n- Today is Monday <!-- expired: 0 -->\n",
        )
        decisions = json.dumps(
            [
                {"fact": "Today is Monday", "decision": "EXPIRED"},
            ]
        )
        client = _make_client(decisions)
        s = _make_summarizer(root, client)

        await s._review_expiration("recent log content")

        client.chat.completions.create.assert_called_once()
        content = s.memory_file.read_text()
        # Tea not mentioned by LLM — unchanged
        assert "User likes tea <!-- expired: 0 -->" in content
        # Monday expired (counter incremented)
        assert "Today is Monday <!-- expired: 1 -->" in content

    @pytest.mark.asyncio
    async def test_skips_when_no_reviewable_facts(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Identity\n- User is Bob\n")
        client = _make_client()
        s = _make_summarizer(root, client)

        await s._review_expiration("log")

        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Facts\n- some fact <!-- expired: 0 -->\n")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
        s = _make_summarizer(root, client)

        # Should not raise
        await s._review_expiration("log")
        assert "some fact" in s.memory_file.read_text()


class TestPromoteWithExpiredMarker:
    """Verify that PROMOTE adds <!-- expired: 0 --> marker."""

    def test_promoted_fact_has_marker(self, tmp_path):
        root = _setup(tmp_path, "# Memory\n\n## Facts\n")
        s = _make_summarizer(root)
        s._apply_decisions(
            [
                {
                    "fact": "New important fact",
                    "category": "FACT",
                    "decision": "PROMOTE",
                }
            ]
        )
        content = s.memory_file.read_text()
        assert "- New important fact <!-- expired: 0 -->" in content
