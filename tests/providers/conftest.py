from pathlib import Path

import pytest

from providers.llm_history_manager import LLMHistoryManager


@pytest.fixture(autouse=True)
def _isolate_llm_history_file(monkeypatch, tmp_path: Path):
    """
    Ensure LLMHistoryManager disk persistence does not leak state across tests.
    Each test gets its own temp history file.
    """
    path = tmp_path / "llm_history.json"
    monkeypatch.setattr(LLMHistoryManager, "_history_file_path", lambda self: str(path))
    yield
