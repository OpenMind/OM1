from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from providers.llm_history_manager import ChatMessage, LLMHistoryManager


def _make_config(history_length: int = 5):
    cfg = MagicMock()
    cfg.model = "gpt-4o"
    cfg.history_length = history_length
    cfg.agent_name = "TestBot"
    return cfg


def _patch_history_path(monkeypatch, path: Path):
    monkeypatch.setattr(
        LLMHistoryManager,
        "_history_file_path",
        lambda self: str(path),
    )


def test_persist_and_restore_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    _patch_history_path(monkeypatch, path)

    cfg = _make_config(history_length=5)
    client = AsyncMock()

    hm1 = LLMHistoryManager(cfg, client)
    hm1.frame_index = 7
    hm1.history = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
    ]
    hm1._save_history_to_disk()
    assert path.exists()

    hm2 = LLMHistoryManager(cfg, client)
    assert hm2.frame_index == 7
    assert [(m.role, m.content) for m in hm2.history] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_corrupt_history_file_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    path.write_text("not-json", encoding="utf-8")
    _patch_history_path(monkeypatch, path)

    cfg = _make_config(history_length=5)
    client = AsyncMock()

    hm = LLMHistoryManager(cfg, client)
    assert hm.history == []


def test_history_length_zero_skips_persistence(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    _patch_history_path(monkeypatch, path)

    cfg = _make_config(history_length=0)
    client = AsyncMock()

    hm = LLMHistoryManager(cfg, client)
    hm.history = [ChatMessage(role="user", content="hello")]
    hm._save_history_to_disk()
    assert not path.exists()
