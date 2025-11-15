from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from backgrounds.base import Background, BackgroundConfig
from backgrounds.orchestrator import BackgroundOrchestrator


class MockBackground(Background):
    def __init__(self, config: BackgroundConfig):
        super().__init__(config)

    def run(self):
        pass


@pytest.fixture
def mock_background():
    background1 = MockBackground(config=BackgroundConfig(name="bg1"))
    background2 = MockBackground(config=BackgroundConfig(name="bg2"))
    return SimpleNamespace(backgrounds=[background1, background2])


@pytest.fixture
def orchestrator(mock_background):
    return BackgroundOrchestrator(mock_background)


def test_background_orchestrator_initialization(mock_background):
    """Test that BackgroundOrchestrator initializes correctly."""
    orchestrator = BackgroundOrchestrator(mock_background)
    assert orchestrator._config == mock_background
    assert orchestrator._background_workers == 2
    assert orchestrator._background_executor is not None


def test_start_background(orchestrator):
    """Test that backgrounds are started in separate threads."""
    try:
        futures = orchestrator.start()

        assert isinstance(orchestrator._background_executor, ThreadPoolExecutor)
        assert orchestrator._background_executor._max_workers == orchestrator._background_workers

        assert len(orchestrator._submitted_backgrounds) == len(
            orchestrator._config.backgrounds
        )
        assert isinstance(futures, dict)

        expected_background_names = {bg.name for bg in orchestrator._config.backgrounds}
        assert orchestrator._submitted_backgrounds == expected_background_names
        assert set(futures) == expected_background_names
        assert all(isinstance(future, Future) for future in futures.values())
    finally:
        orchestrator.stop()


def test_start_without_configured_backgrounds():
    config = SimpleNamespace(backgrounds=None)
    orchestrator = BackgroundOrchestrator(config)

    assert orchestrator._background_workers == 0
    assert orchestrator._background_executor is None
    assert orchestrator.start() == {}
    assert orchestrator._background_executor is None

    orchestrator.stop()
