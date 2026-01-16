import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

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
    config = Mock(spec=BackgroundConfig)
    background1 = MockBackground(config=BackgroundConfig())
    background1.name = "bg1"
    background2 = MockBackground(config=BackgroundConfig())
    background2.name = "bg2"
    config.backgrounds = [background1, background2]
    return config


@pytest.fixture
def orchestrator(mock_background):
    return BackgroundOrchestrator(mock_background)


@pytest.mark.asyncio
async def test_background_orchestrator_initialization(mock_background):
    """Test that BackgroundOrchestrator initializes correctly."""
    orchestrator = BackgroundOrchestrator(mock_background)
    assert orchestrator._config == mock_background
    assert orchestrator._background_workers == 2


@pytest.mark.asyncio
async def test_start_background(orchestrator):
    """Test that backgrounds are started in separate threads."""
    try:
        future = orchestrator.start()

        assert isinstance(orchestrator._background_executor, ThreadPoolExecutor)
        assert (
            orchestrator._background_executor._max_workers
            == orchestrator._background_workers
        )

        assert len(orchestrator._submitted_backgrounds) == len(
            orchestrator._config.backgrounds
        )
        assert isinstance(future, asyncio.Future)

        expected_background_names = {bg.name for bg in orchestrator._config.backgrounds}
        assert orchestrator._submitted_backgrounds == expected_background_names
    finally:
        orchestrator.stop()
