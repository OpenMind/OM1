import asyncio

import pytest


@pytest.fixture
def event_loop():
    """Create a fresh event loop for each test to prevent task leakage"""
    loop = asyncio.new_event_loop()
    yield loop
    # Cancel all pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


@pytest.fixture(autouse=True)
def mock_history_path(tmp_path, monkeypatch):
    """Redirect conversation history file to a temporary directory during tests."""
    test_history_file = tmp_path / "conversation_history.json"
    monkeypatch.setattr("src.providers.history_storage.HISTORY_FILE", test_history_file)
