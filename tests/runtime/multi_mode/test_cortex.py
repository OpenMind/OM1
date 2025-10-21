from unittest.mock import AsyncMock, Mock, patch

import pytest

from runtime.multi_mode.config import ModeConfig, ModeSystemConfig
from runtime.multi_mode.cortex import ModeCortexRuntime


@pytest.fixture
def sample_mode_config():
    mode_config = ModeConfig(
        name="test_mode",
        display_name="Test Mode",
        description="A test mode",
        system_prompt_base="You are a test agent",
    )
    return mode_config


@pytest.fixture
def mock_mode_config():
    """Mock mode config for testing."""
    mock_config = Mock(spec=ModeConfig)
    mock_config.name = "test_mode"
    mock_config.display_name = "Test Mode"
    mock_config.description = "A test mode"
    mock_config.system_prompt_base = "You are a test agent"
    mock_config.load_components = Mock()
    mock_config.to_runtime_config = Mock()
    return mock_config


@pytest.fixture
def mock_system_config(mock_mode_config):
    """Mock system configuration for testing."""
    config = Mock(spec=ModeSystemConfig)
    config.name = "test_system"
    config.default_mode = "default"
    config.modes = {
        "default": mock_mode_config,
        "advanced": mock_mode_config,
    }
    return config


@pytest.fixture
def mock_mode_manager():
    """Mock mode manager for testing."""
    manager = Mock()
    manager.current_mode_name = "default"
    manager.add_transition_callback = Mock()
    manager.process_tick = AsyncMock(return_value=None)
    return manager


@pytest.fixture
def mock_orchestrators():
    """Mock orchestrators for testing."""
    return {
        "fuser": Mock(),
        "action_orchestrator": Mock(),
        "simulator_orchestrator": Mock(),
        "background_orchestrator": Mock(),
        "input_orchestrator": Mock(),
    }


@pytest.fixture
def cortex_runtime(mock_system_config):
    """ModeCortexRuntime instance for testing."""
    with (
        patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
        patch("runtime.multi_mode.cortex.IOProvider") as mock_io_provider_class,
        patch(
            "runtime.multi_mode.cortex.SleepTickerProvider"
        ) as mock_sleep_provider_class,
    ):
        mock_manager = Mock()
        mock_manager.current_mode_name = "default"
        mock_manager.add_transition_callback = Mock()
        mock_manager_class.return_value = mock_manager

        mock_io_provider = Mock()
        mock_io_provider_class.return_value = mock_io_provider

        mock_sleep_provider = Mock()
        mock_sleep_provider.skip_sleep = False
        mock_sleep_provider_class.return_value = mock_sleep_provider

        runtime = ModeCortexRuntime(mock_system_config)
        runtime.mode_manager = mock_manager
        runtime.io_provider = mock_io_provider
        runtime.sleep_ticker_provider = mock_sleep_provider

        return runtime, {
            "mode_manager": mock_manager,
            "io_provider": mock_io_provider,
            "sleep_provider": mock_sleep_provider,
        }


class TestModeCortexRuntime:
    """Test cases for ModeCortexRuntime class."""

    def test_initialization(self, mock_system_config):
        """Test cortex runtime initialization."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config)

            assert runtime.mode_config == mock_system_config
            assert runtime.current_config is None
            assert runtime.fuser is None
            assert runtime.action_orchestrator is None
            assert runtime.simulator_orchestrator is None
            assert runtime.background_orchestrator is None
            assert runtime.input_orchestrator is None
            assert runtime._mode_initialized is False

            mock_manager.add_transition_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mode(self, cortex_runtime, mock_mode_config):
        """Test mode initialization."""
        runtime, mocks = cortex_runtime

        with (
            patch("runtime.multi_mode.cortex.Fuser") as mock_fuser_class,
            patch("runtime.multi_mode.cortex.ActionOrchestrator") as mock_action_class,
            patch(
                "runtime.multi_mode.cortex.SimulatorOrchestrator"
            ) as mock_simulator_class,
            patch(
                "runtime.multi_mode.cortex.BackgroundOrchestrator"
            ) as mock_background_class,
        ):
            mock_fuser = Mock()
            mock_action_orch = Mock()
            mock_simulator_orch = Mock()
            mock_background_orch = Mock()

            mock_fuser_class.return_value = mock_fuser
            mock_action_class.return_value = mock_action_orch
            mock_simulator_class.return_value = mock_simulator_orch
            mock_background_class.return_value = mock_background_orch

            runtime.mode_config.modes = {"test_mode": mock_mode_config}

            await runtime._initialize_mode("test_mode")

            mock_mode_config.load_components.assert_called_once_with(
                runtime.mode_config
            )
            mock_mode_config.to_runtime_config.assert_called_once_with(
                runtime.mode_config
            )

            assert runtime.fuser == mock_fuser
            assert runtime.action_orchestrator == mock_action_orch
            assert runtime.simulator_orchestrator == mock_simulator_orch
            assert runtime.background_orchestrator == mock_background_orch

    @pytest.mark.asyncio
    async def test_on_mode_transition(self, cortex_runtime):
        """Test mode transition handling."""
        runtime, mocks = cortex_runtime

        with (
            patch.object(runtime, "_stop_current_orchestrators") as mock_stop,
            patch.object(runtime, "_initialize_mode") as mock_init,
            patch.object(runtime, "_start_orchestrators") as mock_start,
        ):
            mock_from_mode = Mock()
            mock_to_mode = Mock()
            runtime.mode_config.modes = {
                "from_mode": mock_from_mode,
                "to_mode": mock_to_mode,
            }

            await runtime._on_mode_transition("from_mode", "to_mode")

            mock_stop.assert_called_once()
            mock_init.assert_called_once_with("to_mode")
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_mode_transition_no_announcement(self, cortex_runtime):
        """Test mode transition without announcement."""
        runtime, mocks = cortex_runtime

        with (
            patch.object(runtime, "_stop_current_orchestrators"),
            patch.object(runtime, "_initialize_mode"),
            patch.object(runtime, "_start_orchestrators"),
        ):
            mock_mode = Mock()
            runtime.mode_config.modes = {"to_mode": mock_mode}

            await runtime._on_mode_transition("from_mode", "to_mode")

    @pytest.mark.asyncio
    async def test_on_mode_transition_exception(self, cortex_runtime):
        """Test mode transition with exception handling."""
        runtime, mocks = cortex_runtime

        mock_from_mode = Mock()
        mock_to_mode = Mock()
        runtime.mode_config.modes = {
            "from_mode": mock_from_mode,
            "to_mode": mock_to_mode,
        }

        with patch.object(
            runtime, "_stop_current_orchestrators", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception, match="Test error"):
                await runtime._on_mode_transition("from_mode", "to_mode")

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators(self, cortex_runtime):
        """Test stopping current orchestrators."""
        runtime, mocks = cortex_runtime

        mock_input_task = Mock()
        mock_input_task.done.return_value = False
        mock_input_task.cancel = Mock()

        mock_simulator_task = Mock()
        mock_simulator_task.done.return_value = False
        mock_simulator_task.cancel = Mock()

        mock_action_task = Mock()
        mock_action_task.done.return_value = False
        mock_action_task.cancel = Mock()

        mock_background_task = Mock()
        mock_background_task.done.return_value = False
        mock_background_task.cancel = Mock()

        runtime.input_listener_task = mock_input_task
        runtime.simulator_task = mock_simulator_task
        runtime.action_task = mock_action_task
        runtime.background_task = mock_background_task

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await runtime._stop_current_orchestrators()

            mock_input_task.cancel.assert_called_once()
            mock_simulator_task.cancel.assert_called_once()
            mock_action_task.cancel.assert_called_once()
            mock_background_task.cancel.assert_called_once()

            mock_gather.assert_called_once()

            assert runtime.input_listener_task is None
            assert runtime.simulator_task is None
            assert runtime.action_task is None
            assert runtime.background_task is None

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators_done_tasks(self, cortex_runtime):
        """Test stopping orchestrators with already done tasks."""
        runtime, mocks = cortex_runtime

        mock_task = Mock()
        mock_task.done.return_value = True
        mock_task.cancel = Mock()

        runtime.input_listener_task = mock_task

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await runtime._stop_current_orchestrators()

            mock_task.cancel.assert_not_called()
            mock_gather.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_orchestrators_no_config(self, cortex_runtime):
        """Test starting orchestrators without current config raises error."""
        runtime, mocks = cortex_runtime
        runtime.current_config = None

        with pytest.raises(RuntimeError, match="No current config available"):
            await runtime._start_orchestrators()

    @pytest.mark.asyncio
    async def test_cleanup_tasks(self, cortex_runtime):
        """Test cleanup of all tasks."""
        runtime, mocks = cortex_runtime

        mock_task1 = Mock()
        mock_task1.done.return_value = False
        mock_task1.cancel = Mock()

        mock_task2 = Mock()
        mock_task2.done.return_value = False
        mock_task2.cancel = Mock()

        runtime.input_listener_task = mock_task1
        runtime.simulator_task = mock_task2

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await runtime._cleanup_tasks()

            mock_task1.cancel.assert_called_once()
            mock_task2.cancel.assert_called_once()
            mock_gather.assert_called_once()
