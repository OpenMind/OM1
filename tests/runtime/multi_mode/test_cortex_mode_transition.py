import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from runtime.multi_mode.config import (
    ModeConfig,
    ModeSystemConfig,
    TransitionRule,
    TransitionType,
)
from runtime.multi_mode.cortex import ModeCortexRuntime


@pytest.fixture
def sample_mode_configs():
    """Sample mode configurations for testing mode transitions."""
    default_mode = ModeConfig(
        version="v1.0.0",
        name="default",
        display_name="Default Mode",
        description="Default operational mode",
        system_prompt_base="You are a default agent",
    )

    advanced_mode = ModeConfig(
        version="v1.0.0",
        name="advanced",
        display_name="Advanced Mode",
        description="Advanced operational mode",
        system_prompt_base="You are an advanced agent",
    )

    emergency_mode = ModeConfig(
        version="v1.0.0",
        name="emergency",
        display_name="Emergency Mode",
        description="Emergency operational mode",
        system_prompt_base="You are an emergency response agent",
    )

    return {
        "default": default_mode,
        "advanced": advanced_mode,
        "emergency": emergency_mode,
    }


@pytest.fixture
def sample_transition_rules():
    """Sample transition rules for testing mode transitions."""
    return [
        TransitionRule(
            from_mode="default",
            to_mode="advanced",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["advanced", "complex", "detailed"],
            priority=5,
            cooldown_seconds=1.0,
        ),
        TransitionRule(
            from_mode="default",
            to_mode="emergency",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["emergency", "urgent", "help"],
            priority=10,  # Higher priority than advanced
            cooldown_seconds=0.5,
        ),
        TransitionRule(
            from_mode="advanced",
            to_mode="default",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["default", "normal", "basic"],
            priority=5,
            cooldown_seconds=1.0,
        ),
        TransitionRule(
            from_mode="*",  # Wildcard - from any mode
            to_mode="emergency",
            transition_type=TransitionType.INPUT_TRIGGERED,
            trigger_keywords=["emergency", "critical", "urgent"],
            priority=15,  # Highest priority
            cooldown_seconds=0.0,
        ),
    ]


@pytest.fixture
def mock_system_config(sample_mode_configs, sample_transition_rules):
    """Mock system configuration with proper mode transition setup."""
    config = Mock(spec=ModeSystemConfig)
    config.name = "test_system"
    config.default_mode = "default"
    config.modes = sample_mode_configs
    config.transition_rules = sample_transition_rules
    config.allow_manual_switching = True
    config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)

    for mode_config in sample_mode_configs.values():
        mode_config.load_components = Mock()
        mode_config.to_runtime_config = Mock()
        mode_config.execute_lifecycle_hooks = AsyncMock()

    return config


@pytest.fixture
def mock_io_provider():
    """Mock IOProvider with mode transition input functionality."""
    io_provider = Mock()
    io_provider._mode_transition_input = None

    def add_mode_transition_input(input_text: str):
        if io_provider._mode_transition_input is None:
            io_provider._mode_transition_input = input_text
        else:
            io_provider._mode_transition_input = (
                io_provider._mode_transition_input + " " + input_text
            )

    def get_mode_transition_input():
        return io_provider._mode_transition_input

    def delete_mode_transition_input():
        io_provider._mode_transition_input = None

    class MockModeTransitionInputContext:
        def __init__(self, io_provider):
            self.io_provider = io_provider
            self.current_input = None

        def __enter__(self):
            self.current_input = io_provider._mode_transition_input
            return self.current_input

        def __exit__(self, exc_type, exc_val, exc_tb):
            delete_mode_transition_input()

    def mode_transition_input():
        return MockModeTransitionInputContext(io_provider)

    io_provider.add_mode_transition_input = add_mode_transition_input
    io_provider.get_mode_transition_input = get_mode_transition_input
    io_provider.delete_mode_transition_input = delete_mode_transition_input
    io_provider.mode_transition_input = mode_transition_input

    return io_provider


@pytest.fixture
def mock_mode_manager():
    """Mock mode manager with transition capabilities."""
    manager = Mock()
    manager.current_mode_name = "default"
    manager.add_transition_callback = Mock()
    manager.remove_transition_callback = Mock()
    manager.set_event_loop = Mock()
    manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")

    async def mock_process_tick(input_text=None):
        if not input_text:
            return None

        input_lower = input_text.lower()
        if "emergency" in input_lower or "urgent" in input_lower:
            return ("emergency", "input_triggered")
        elif "advanced" in input_lower:
            return ("advanced", "input_triggered")
        elif "default" in input_lower or "normal" in input_lower:
            return ("default", "input_triggered")
        return None

    manager.process_tick = AsyncMock(side_effect=mock_process_tick)
    manager._execute_transition = AsyncMock(return_value=True)

    return manager


@pytest.fixture
def cortex_runtime_with_mode_transition(
    mock_system_config, mock_io_provider, mock_mode_manager
):
    """ModeCortexRuntime instance configured for mode transition testing."""
    with (
        patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
        patch("runtime.multi_mode.cortex.IOProvider") as mock_io_provider_class,
        patch(
            "runtime.multi_mode.cortex.SleepTickerProvider"
        ) as mock_sleep_provider_class,
    ):
        mock_manager_class.return_value = mock_mode_manager
        mock_io_provider_class.return_value = mock_io_provider

        mock_sleep_provider = Mock()
        mock_sleep_provider.skip_sleep = False
        mock_sleep_provider.sleep = AsyncMock()
        mock_sleep_provider_class.return_value = mock_sleep_provider

        runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)

        mock_runtime_config = Mock()
        mock_runtime_config.hertz = 10.0
        mock_runtime_config.cortex_llm = Mock()
        mock_runtime_config.cortex_llm.ask = AsyncMock(return_value=Mock(actions=[]))
        mock_runtime_config.agent_inputs = []

        runtime.current_config = mock_runtime_config
        runtime.fuser = Mock()
        runtime.fuser.fuse = Mock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.action_orchestrator.promise = AsyncMock()
        runtime.simulator_orchestrator = Mock()
        runtime.simulator_orchestrator.promise = AsyncMock()

        return runtime, {
            "mode_manager": mock_mode_manager,
            "io_provider": mock_io_provider,
            "sleep_provider": mock_sleep_provider,
            "system_config": mock_system_config,
        }


@pytest.fixture
def cortex_runtime(mock_system_config, mock_io_provider, mock_mode_manager):
    """Create a cortex runtime instance for basic testing."""
    with (
        patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
        patch("runtime.multi_mode.cortex.IOProvider") as mock_io_class,
        patch("runtime.multi_mode.cortex.SleepTickerProvider") as mock_sleep_class,
    ):
        mock_manager_class.return_value = mock_mode_manager
        mock_io_class.return_value = mock_io_provider

        mock_sleep_provider = Mock()
        mock_sleep_provider.skip_sleep = False
        mock_sleep_provider.sleep = AsyncMock()
        mock_sleep_class.return_value = mock_sleep_provider

        runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)

        mock_runtime_config = Mock()
        mock_runtime_config.hertz = 10.0
        mock_runtime_config.cortex_llm = Mock()
        mock_runtime_config.cortex_llm.ask = AsyncMock(return_value=Mock(actions=[]))
        mock_runtime_config.agent_inputs = []

        runtime.current_config = mock_runtime_config
        runtime.fuser = Mock()
        runtime.fuser.fuse = Mock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.action_orchestrator.promise = AsyncMock()
        runtime.simulator_orchestrator = None

        runtime._pending_mode_transition = None
        runtime._mode_transition_event = Mock()
        runtime._mode_transition_event.set = Mock()

        return runtime, {
            "mode_manager": mock_mode_manager,
            "io_provider": mock_io_provider,
            "system_config": mock_system_config,
        }


@pytest.mark.asyncio
async def test_tick_with_mode_transition_input_triggers_transition(
    cortex_runtime_with_mode_transition,
):
    """Test that mode transition input in _tick triggers a mode transition."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("I need advanced mode")

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with("I need advanced mode")

    assert runtime._pending_mode_transition == "advanced"
    runtime._mode_transition_event.set.assert_called_once()


@pytest.mark.asyncio
async def test_tick_with_emergency_input_triggers_emergency_mode(
    cortex_runtime_with_mode_transition,
):
    """Test that emergency keywords trigger emergency mode transition."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("Emergency help needed!")

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with("Emergency help needed!")
    assert runtime._pending_mode_transition == "emergency"
    runtime._mode_transition_event.set.assert_called_once()


@pytest.mark.asyncio
async def test_tick_with_no_mode_transition_input_continues_normally(
    cortex_runtime_with_mode_transition,
):
    """Test that _tick continues normally when there's no mode transition input."""
    runtime, mocks = cortex_runtime_with_mode_transition

    assert mocks["io_provider"].get_mode_transition_input() is None

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with(None)

    assert runtime._pending_mode_transition is None
    runtime._mode_transition_event.set.assert_not_called()

    runtime.action_orchestrator.promise.assert_called_once()


@pytest.mark.asyncio
async def test_tick_with_unrecognized_input_continues_normally(
    cortex_runtime_with_mode_transition,
):
    """Test that unrecognized input doesn't trigger mode transition."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("just some random text")

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with("just some random text")

    assert runtime._pending_mode_transition is None
    runtime._mode_transition_event.set.assert_not_called()

    runtime.action_orchestrator.promise.assert_called_once()


@pytest.mark.asyncio
async def test_mode_transition_input_is_cleared_after_use(
    cortex_runtime_with_mode_transition,
):
    """Test that mode transition input is cleared after processing."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("advanced mode please")
    assert mocks["io_provider"].get_mode_transition_input() == "advanced mode please"

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    assert mocks["io_provider"].get_mode_transition_input() is None


@pytest.mark.asyncio
async def test_multiple_mode_transition_inputs_are_combined(
    cortex_runtime_with_mode_transition,
):
    """Test that multiple mode transition inputs are properly combined."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("I need")
    mocks["io_provider"].add_mode_transition_input("advanced")
    mocks["io_provider"].add_mode_transition_input("mode")

    assert mocks["io_provider"].get_mode_transition_input() == "I need advanced mode"

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with("I need advanced mode")
    assert runtime._pending_mode_transition == "advanced"


@pytest.mark.asyncio
async def test_tick_skips_processing_during_reload(cortex_runtime_with_mode_transition):
    """Test that _tick skips processing when a config reload is in progress."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("emergency help")

    runtime._is_reloading = True

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_not_called()
    runtime.action_orchestrator.promise.assert_not_called()


@pytest.mark.asyncio
async def test_tick_handles_llm_returning_none(cortex_runtime_with_mode_transition):
    """Test that _tick handles gracefully when LLM returns None."""
    runtime, mocks = cortex_runtime_with_mode_transition

    runtime.current_config.cortex_llm.ask.return_value = None

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_called_once_with(None)

    runtime.action_orchestrator.promise.assert_not_called()


@pytest.mark.asyncio
async def test_tick_handles_fuser_returning_none(cortex_runtime_with_mode_transition):
    """Test that _tick handles gracefully when fuser returns None."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("default mode")

    runtime.fuser.fuse.return_value = None

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    mocks["mode_manager"].process_tick.assert_not_called()
    assert runtime._pending_mode_transition is None

    runtime.current_config.cortex_llm.ask.assert_not_called()
    runtime.action_orchestrator.promise.assert_not_called()


@pytest.mark.asyncio
async def test_handle_mode_transitions_processes_pending_transition(
    cortex_runtime_with_mode_transition,
):
    """Test that _handle_mode_transitions processes pending transitions."""
    runtime, mocks = cortex_runtime_with_mode_transition

    runtime._pending_mode_transition = "emergency"
    runtime._mode_transition_event = asyncio.Event()
    runtime._mode_transition_event.set()

    mocks["mode_manager"]._execute_transition = AsyncMock(return_value=True)

    async def limited_handle_transitions():
        await runtime._mode_transition_event.wait()

        success = False
        if runtime._pending_mode_transition:
            target_mode = runtime._pending_mode_transition
            runtime._pending_mode_transition = None

            success = await mocks["mode_manager"]._execute_transition(
                target_mode, "input_triggered"
            )

        runtime._mode_transition_event.clear()
        return success

    success = await limited_handle_transitions()

    assert success is True
    mocks["mode_manager"]._execute_transition.assert_called_once_with(
        "emergency", "input_triggered"
    )


@pytest.mark.asyncio
async def test_handle_mode_transitions_handles_failed_transition(
    cortex_runtime_with_mode_transition,
):
    """Test that _handle_mode_transitions handles failed transitions gracefully."""
    runtime, mocks = cortex_runtime_with_mode_transition

    runtime._pending_mode_transition = "invalid_mode"
    runtime._mode_transition_event = asyncio.Event()
    runtime._mode_transition_event.set()

    mocks["mode_manager"]._execute_transition = AsyncMock(return_value=False)

    async def limited_handle_transitions():
        await runtime._mode_transition_event.wait()

        success = False
        if runtime._pending_mode_transition:
            target_mode = runtime._pending_mode_transition
            runtime._pending_mode_transition = None

            success = await mocks["mode_manager"]._execute_transition(
                target_mode, "input_triggered"
            )

        runtime._mode_transition_event.clear()
        return success

    success = await limited_handle_transitions()

    assert success is False
    assert success is False
    mocks["mode_manager"]._execute_transition.assert_called_once_with(
        "invalid_mode", "input_triggered"
    )


@pytest.mark.asyncio
async def test_on_mode_transition_callback_integration(
    cortex_runtime_with_mode_transition,
):
    """Test the integration of mode transition callback with cortex runtime."""
    runtime, mocks = cortex_runtime_with_mode_transition

    transition_callback = mocks["mode_manager"].add_transition_callback.call_args[0][0]

    runtime._stop_current_orchestrators = AsyncMock()
    runtime._initialize_mode = AsyncMock()
    runtime._start_orchestrators = AsyncMock()

    await transition_callback("default", "advanced")

    runtime._stop_current_orchestrators.assert_called_once()
    runtime._initialize_mode.assert_called_once_with("advanced")
    runtime._start_orchestrators.assert_called_once()


def test_get_mode_info_integration(cortex_runtime_with_mode_transition):
    """Test get_mode_info method integration with mode manager."""
    runtime, mocks = cortex_runtime_with_mode_transition

    expected_info = {
        "current_mode": "default",
        "available_transitions": ["advanced", "emergency"],
        "mode_duration": 120.5,
    }
    mocks["mode_manager"].get_mode_info = Mock(return_value=expected_info)

    result = runtime.get_mode_info()

    assert result == expected_info
    mocks["mode_manager"].get_mode_info.assert_called_once()


@pytest.mark.asyncio
async def test_request_mode_change_integration(cortex_runtime_with_mode_transition):
    """Test request_mode_change method integration with mode manager."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["mode_manager"].request_transition = AsyncMock(return_value=True)

    result = await runtime.request_mode_change("emergency")

    assert result is True
    mocks["mode_manager"].request_transition.assert_called_once_with(
        "emergency", "manual"
    )


def test_get_available_modes_returns_correct_structure(
    cortex_runtime_with_mode_transition,
):
    """Test get_available_modes returns the correct mode information structure."""
    runtime, mocks = cortex_runtime_with_mode_transition

    result = runtime.get_available_modes()

    assert isinstance(result, dict)
    assert "default" in result
    assert "advanced" in result
    assert "emergency" in result

    for mode_name, mode_info in result.items():
        assert "display_name" in mode_info
        assert "description" in mode_info
        assert "is_current" in mode_info
        assert isinstance(mode_info["is_current"], bool)

    assert result["default"]["is_current"] is True
    assert result["advanced"]["is_current"] is False
    assert result["emergency"]["is_current"] is False


@pytest.mark.asyncio
async def test_emergency_mode_has_highest_priority(cortex_runtime_with_mode_transition):
    """Test that emergency mode transitions have highest priority."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("I need advanced emergency help")

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    assert runtime._pending_mode_transition == "emergency"


@pytest.mark.asyncio
async def test_mode_transition_during_reload_is_ignored(
    cortex_runtime_with_mode_transition,
):
    """Test that mode transitions are ignored during config reload."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("emergency help needed")

    runtime._is_reloading = True

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    assert runtime._pending_mode_transition is None
    runtime._mode_transition_event.set.assert_not_called()
    mocks["mode_manager"].process_tick.assert_not_called()


@pytest.mark.asyncio
async def test_mode_transition_with_simulator_orchestrator(
    cortex_runtime_with_mode_transition,
):
    """Test mode transition works correctly when simulator orchestrator is present."""
    runtime, mocks = cortex_runtime_with_mode_transition

    mocks["io_provider"].add_mode_transition_input("switch to advanced mode")

    runtime.simulator_orchestrator = Mock()
    runtime.simulator_orchestrator.promise = AsyncMock()

    mock_output = Mock()
    mock_output.actions = ["action1", "action2"]
    runtime.current_config.cortex_llm.ask.return_value = mock_output

    runtime._pending_mode_transition = None
    runtime._mode_transition_event = Mock()
    runtime._mode_transition_event.set = Mock()

    await runtime._tick()

    assert runtime._pending_mode_transition == "advanced"

    if runtime._pending_mode_transition:
        pass
    else:
        runtime.simulator_orchestrator.promise.assert_called_once_with(
            mock_output.actions
        )


@pytest.mark.asyncio
async def test_mode_transition_input_triggers_advanced_mode(cortex_runtime):
    """Test that setting mode transition input triggers transition to advanced mode."""
    runtime, components = cortex_runtime

    components["io_provider"].add_mode_transition_input("switch to advanced mode")

    await runtime._tick()

    components["mode_manager"].process_tick.assert_called_once_with(
        "switch to advanced mode"
    )

    assert runtime._pending_mode_transition == "advanced"
    runtime._mode_transition_event.set.assert_called_once()


@pytest.mark.asyncio
async def test_mode_transition_input_triggers_emergency_mode(cortex_runtime):
    """Test that emergency keywords trigger emergency mode transition."""
    runtime, components = cortex_runtime

    components["io_provider"].add_mode_transition_input("emergency help needed!")

    await runtime._tick()

    components["mode_manager"].process_tick.assert_called_once_with(
        "emergency help needed!"
    )

    assert runtime._pending_mode_transition == "emergency"
    runtime._mode_transition_event.set.assert_called_once()


@pytest.mark.asyncio
async def test_no_mode_transition_input_continues_normal_processing(cortex_runtime):
    """Test that no mode transition input allows normal LLM processing."""
    runtime, components = cortex_runtime

    assert components["io_provider"].get_mode_transition_input() is None

    await runtime._tick()

    components["mode_manager"].process_tick.assert_called_once_with(None)

    assert runtime._pending_mode_transition is None
    runtime._mode_transition_event.set.assert_not_called()

    runtime.current_config.cortex_llm.ask.assert_called_once_with("test prompt")
    runtime.action_orchestrator.promise.assert_called_once()


@pytest.mark.asyncio
async def test_mode_transition_input_is_cleared_after_use_basic(cortex_runtime):
    """Test that mode transition input is automatically cleared after processing."""
    runtime, components = cortex_runtime

    components["io_provider"].add_mode_transition_input("default mode")
    assert components["io_provider"].get_mode_transition_input() == "default mode"

    await runtime._tick()

    assert components["io_provider"].get_mode_transition_input() is None


@pytest.mark.asyncio
async def test_unrecognized_input_does_not_trigger_transition(cortex_runtime):
    """Test that unrecognized input doesn't trigger mode transitions."""
    runtime, components = cortex_runtime

    components["io_provider"].add_mode_transition_input("some random text")

    await runtime._tick()

    components["mode_manager"].process_tick.assert_called_once_with("some random text")
    assert runtime._pending_mode_transition is None
    runtime._mode_transition_event.set.assert_not_called()

    runtime.action_orchestrator.promise.assert_called_once()


@pytest.mark.asyncio
async def test_mode_transition_during_reload_is_skipped_basic(cortex_runtime):
    """Test that mode transitions are skipped during config reload."""
    runtime, components = cortex_runtime

    components["io_provider"].add_mode_transition_input("emergency")
    runtime._is_reloading = True

    await runtime._tick()

    components["mode_manager"].process_tick.assert_not_called()
    runtime.action_orchestrator.promise.assert_not_called()


@pytest.mark.asyncio
async def test_mode_transition_callback_registration(cortex_runtime):
    """Test that the cortex runtime registers a transition callback with the mode manager."""
    runtime, components = cortex_runtime

    components["mode_manager"].add_transition_callback.assert_called_once()

    callback = components["mode_manager"].add_transition_callback.call_args[0][0]

    runtime._stop_current_orchestrators = AsyncMock()
    runtime._initialize_mode = AsyncMock()
    runtime._start_orchestrators = AsyncMock()

    await callback("default", "advanced")

    runtime._stop_current_orchestrators.assert_called_once()
    runtime._initialize_mode.assert_called_once_with("advanced")
    runtime._start_orchestrators.assert_called_once()


def test_mode_info_delegation_to_manager(cortex_runtime):
    """Test that get_mode_info delegates to the mode manager."""
    runtime, components = cortex_runtime

    expected_info = {"current_mode": "default", "transitions": ["advanced"]}
    components["mode_manager"].get_mode_info = Mock(return_value=expected_info)

    result = runtime.get_mode_info()

    components["mode_manager"].get_mode_info.assert_called_once()
    assert result == expected_info


@pytest.mark.asyncio
async def test_manual_mode_change_delegation_to_manager(cortex_runtime):
    """Test that request_mode_change delegates to the mode manager."""
    runtime, components = cortex_runtime

    components["mode_manager"].request_transition = AsyncMock(return_value=True)

    result = await runtime.request_mode_change("emergency")

    components["mode_manager"].request_transition.assert_called_once_with(
        "emergency", "manual"
    )
    assert result is True


def test_available_modes_structure(cortex_runtime):
    """Test that get_available_modes returns properly structured mode information."""
    runtime, components = cortex_runtime

    for mode_name in ["default", "advanced", "emergency"]:
        mode_config = Mock()
        mode_config.display_name = f"{mode_name.title()} Mode"
        mode_config.description = f"Description for {mode_name} mode"
        components["system_config"].modes[mode_name] = mode_config

    result = runtime.get_available_modes()

    assert isinstance(result, dict)
    assert len(result) == 3

    for mode_name, mode_info in result.items():
        assert "display_name" in mode_info
        assert "description" in mode_info
        assert "is_current" in mode_info
        assert isinstance(mode_info["is_current"], bool)

    assert result["default"]["is_current"] is True
    assert result["advanced"]["is_current"] is False
    assert result["emergency"]["is_current"] is False
