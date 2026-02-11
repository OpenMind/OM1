"""Pytest configuration and shared fixtures for all tests."""

import pytest


@pytest.fixture(autouse=True)
def mock_history_path(tmp_path, monkeypatch):
    """Redirect conversation history file to a temporary directory during tests.

    This prevents tests from reading or writing to the real
    data/conversation_history.json file on disk, ensuring test isolation.
    pytest automatically cleans up tmp_path after each test.
    """
    test_history_file = tmp_path / "conversation_history.json"
    monkeypatch.setattr("src.providers.history_storage.HISTORY_FILE", test_history_file)
