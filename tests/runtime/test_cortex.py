import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm.output_model import Action
from runtime.config import RuntimeConfig
from runtime.cortex import CortexRuntime


@pytest.fixture
def mock_config():
    config = Mock(spec=RuntimeConfig, hertz=10.0)
    config.name = "test_config"
    config.cortex_llm = Mock()
    config.agent_inputs = []
    return config


@pytest.fixture
def mock_dependencies():
    return {
        "fuser": Mock(),
        "action_orchestrator": Mock(),
        "simulator_orchestrator": Mock(),
        "background_orchestrator": Mock(),
        "sleep_ticker_provider": Mock(),
        "input_orchestrator": Mock(),
    }


@pytest.fixture
def runtime(mock_config, mock_dependencies):
    with (
        patch("runtime.cortex.Fuser", return_value=mock_dependencies["fuser"]),
        patch(
            "runtime.cortex.ActionOrchestrator",
            return_value=mock_dependencies["action_orchestrator"],
        ),
        patch(
            "runtime.cortex.SimulatorOrchestrator",
            return_value=mock_dependencies["simulator_orchestrator"],
        ),
        patch(
            "runtime.cortex.SleepTickerProvider",
            return_value=mock_dependencies["sleep_ticker_provider"],
        ),
        patch(
            "runtime.cortex.BackgroundOrchestrator",
            return_value=mock_dependencies["background_orchestrator"],
        ),
    ):
        return CortexRuntime(mock_config), mock_dependencies


@pytest.mark.asyncio
async def test_tick_successful_execution(runtime):
    cortex_runtime, mocks = runtime

    # Mock successful flow
    finished_promises = ["promise1"]
    mocks["action_orchestrator"].flush_promises = AsyncMock(
        return_value=(finished_promises, None)
    )
    mocks["fuser"].fuse.return_value = "test prompt"

    action = Action(type="action1", value="val1")

    mock_output = Mock()
    mock_output.actions = [action]
    cortex_runtime.config.cortex_llm.ask = AsyncMock(return_value=mock_output)

    mocks["simulator_orchestrator"].promise = AsyncMock()
    mocks["action_orchestrator"].promise = AsyncMock()
    mocks["background_orchestrator"].promise = AsyncMock()

    await cortex_runtime._tick()

    # Verify flow
    mocks["action_orchestrator"].flush_promises.assert_called_once()
    mocks["fuser"].fuse.assert_called_once_with(
        cortex_runtime.config.agent_inputs, finished_promises
    )
    cortex_runtime.config.cortex_llm.ask.assert_called_once_with("test prompt")
    mocks["simulator_orchestrator"].promise.assert_called_once_with([action])
    mocks["action_orchestrator"].promise.assert_called_once_with([action])


@pytest.mark.asyncio
async def test_tick_no_prompt(runtime):
    cortex_runtime, mocks = runtime

    mocks["action_orchestrator"].flush_promises = AsyncMock(return_value=([], None))
    mocks["fuser"].fuse.return_value = None

    await cortex_runtime._tick()

    cortex_runtime.config.cortex_llm.ask.assert_not_called()
    mocks["simulator_orchestrator"].promise.assert_not_called()
    mocks["action_orchestrator"].promise.assert_not_called()
    mocks["background_orchestrator"].promise.assert_not_called()


@pytest.mark.asyncio
async def test_tick_no_llm_output(runtime):
    cortex_runtime, mocks = runtime

    mocks["action_orchestrator"].flush_promises = AsyncMock(
        return_value=(["promise"], None)
    )
    mocks["fuser"].fuse.return_value = "test prompt"
    cortex_runtime.config.cortex_llm.ask = AsyncMock(return_value=None)

    await cortex_runtime._tick()

    mocks["simulator_orchestrator"].promise.assert_not_called()
    mocks["action_orchestrator"].promise.assert_not_called()
    mocks["background_orchestrator"].promise.assert_not_called()


@pytest.mark.asyncio
async def test_run_cortex_loop(runtime):
    cortex_runtime, mocks = runtime

    # Setup mock for _tick
    cortex_runtime._tick = AsyncMock()
    mocks["sleep_ticker_provider"].skip_sleep = False
    mocks["sleep_ticker_provider"].sleep = AsyncMock()

    # Run loop for 3 iterations then raise exception to stop
    async def side_effect(*args):
        if cortex_runtime._tick.call_count >= 3:
            raise Exception("Stop loop")

    cortex_runtime._tick.side_effect = side_effect

    with pytest.raises(Exception, match="Stop loop"):
        await cortex_runtime._run_cortex_loop()

    assert cortex_runtime._tick.call_count == 3
    assert mocks["sleep_ticker_provider"].sleep.call_count == 3


@pytest.mark.asyncio
async def test_start_input_listeners(runtime):
    cortex_runtime, mocks = runtime

    with patch(
        "runtime.cortex.InputOrchestrator", return_value=mocks["input_orchestrator"]
    ):
        mocks["input_orchestrator"].listen = AsyncMock()
        task = await cortex_runtime._start_input_listeners()

        assert isinstance(task, asyncio.Task)
        mocks["input_orchestrator"].listen.assert_called_once()


@pytest.mark.asyncio
async def test_run_full_runtime(runtime):
    cortex_runtime, _ = runtime

    input_listener_task = asyncio.create_task(asyncio.sleep(0))
    cortex_runtime._start_input_listeners = AsyncMock(return_value=input_listener_task)

    cortex_loop_task = asyncio.create_task(asyncio.sleep(0))
    cortex_runtime._run_cortex_loop = AsyncMock(return_value=cortex_loop_task)

    await cortex_runtime.run()

    cortex_runtime._start_input_listeners.assert_called_once()
    cortex_runtime._run_cortex_loop.assert_called_once()
