import asyncio
from unittest.mock import AsyncMock

import pytest

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from inputs.orchestrator import InputOrchestrator


class MockInput(FuserInput[SensorConfig, str]):
    def __init__(self):
        super().__init__(SensorConfig())
        self.poll_count = 0
        self.max_polls = 3

    async def _listen_loop(self):
        while self.poll_count < self.max_polls:
            await asyncio.sleep(0.1)
            self.poll_count += 1
            yield str(self.poll_count)

    async def raw_to_text(self, raw_input):
        pass

    def formatted_latest_buffer(self):
        return None


class ErrorInput(MockInput):
    async def _listen_loop(self):
        while True:
            await asyncio.sleep(0.1)
            yield "error"
            raise ValueError("Test error")


@pytest.mark.asyncio
async def test_input_orchestrator_initialization():
    """Test that the InputOrchestrator initializes with a list of inputs."""
    inputs = [MockInput(), MockInput()]
    orchestrator = InputOrchestrator(inputs)
    assert orchestrator.inputs == inputs


@pytest.mark.asyncio
async def test_listen_to_input():
    """Test that the InputOrchestrator listens to a single input."""
    mock_input = MockInput()
    mock_input.raw_to_text = AsyncMock()
    orchestrator = InputOrchestrator([mock_input])
    await asyncio.wait_for(orchestrator._listen_to_input(mock_input), timeout=5.0)
    assert mock_input.raw_to_text.call_count == 3


@pytest.mark.asyncio
async def test_listen_multiple_inputs():
    """Test that the InputOrchestrator listens to multiple inputs concurrently."""
    inputs = [MockInput(), MockInput()]
    for input in inputs:
        input.raw_to_text = AsyncMock()
    orchestrator = InputOrchestrator(inputs)
    await asyncio.wait_for(orchestrator.listen(), timeout=5.0)
    for input in inputs:
        assert input.raw_to_text.call_count == 3  # type: ignore


@pytest.mark.asyncio
async def test_input_exception_handling():
    """Test that the InputOrchestrator handles exceptions from inputs."""
    error_input = ErrorInput()
    normal_input = MockInput()
    normal_input.raw_to_text = AsyncMock()
    orchestrator = InputOrchestrator([error_input, normal_input])
    with pytest.raises(ValueError):
        await orchestrator.listen()
