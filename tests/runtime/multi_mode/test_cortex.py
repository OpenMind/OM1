import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from runtime.multi_mode.config import ModeConfig, ModeSystemConfig
from runtime.multi_mode.cortex import (
    MODE_HOT_RELOAD_SAFE_FIELDS,
    MODE_LEVEL_SAFE_FIELDS,
    ModeCortexRuntime,
)


@pytest.fixture
def sample_mode_config():
    mode_config = ModeConfig(
        version="v1.0.2",
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
        mock_manager._get_runtime_config_path = Mock(
            return_value="/fake/path/test_config.json5"
        )
        mock_manager_class.return_value = mock_manager

        mock_io_provider = Mock()
        mock_io_provider_class.return_value = mock_io_provider

        mock_sleep_provider = Mock()
        mock_sleep_provider.skip_sleep = False
        mock_sleep_provider_class.return_value = mock_sleep_provider

        runtime = ModeCortexRuntime(mock_system_config, "test_config")
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
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")

            assert runtime.mode_config == mock_system_config
            assert runtime.mode_config_name == "test_config"
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

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (
                {
                    mock_input_task,
                    mock_simulator_task,
                    mock_action_task,
                    mock_background_task,
                },
                set(),
            )

            await runtime._stop_current_orchestrators()

            mock_input_task.cancel.assert_called_once()
            mock_simulator_task.cancel.assert_called_once()
            mock_action_task.cancel.assert_called_once()
            mock_background_task.cancel.assert_called_once()

            mock_wait.assert_called_once()

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


class TestModeCortexRuntimeHotReload:
    """Test cases for hot reload functionality in ModeCortexRuntime."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing hot reload."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            f.write('{"test": "config"}')
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_hot_reload_initialization_enabled(self, mock_system_config):
        """Test hot reload initialization when enabled."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True, check_interval=30
            )

            assert runtime.hot_reload is True
            assert runtime.check_interval == 30
            assert runtime.last_modified == 1234567890.0
            assert runtime.config_path.endswith("test_config.json5")

    def test_hot_reload_initialization_disabled(self, mock_system_config):
        """Test hot reload initialization when disabled."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=False
            )

            assert runtime.hot_reload is False
            assert runtime.last_modified is None

    def test_get_file_mtime_existing_file(self, mock_system_config, temp_config_file):
        """Test getting modification time of existing file."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.config_path = temp_config_file

            mtime = runtime._get_file_mtime()
            assert mtime > 0

    def test_get_file_mtime_nonexistent_file(self, mock_system_config):
        """Test getting modification time of non-existent file."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.config_path = "/nonexistent/file.json5"

            mtime = runtime._get_file_mtime()
            assert mtime == 0.0

    @pytest.mark.asyncio
    async def test_check_config_changes_file_changed(
        self, mock_system_config, temp_config_file
    ):
        """Test config change detection when file is modified."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.config_path = temp_config_file
            runtime.last_modified = 1.0

            runtime._reload_config = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._reload_config.assert_called_once()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_check_config_changes_no_change(self, mock_system_config):
        """Test config change detection when file is not modified."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.last_modified = 1234567890.0

            runtime._reload_config = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._reload_config.assert_not_called()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_check_config_changes_nonexistent_file(self, mock_system_config):
        """Test config change detection with non-existent file."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.config_path = "/nonexistent/file.json5"
            runtime.last_modified = 1.0

            runtime._reload_config = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._reload_config.assert_not_called()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_reload_config_success(self, mock_system_config):
        """Test successful config reload with unsafe field change."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
            patch("runtime.multi_mode.cortex.load_mode_config") as mock_load_config,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.default_mode = "test_mode"
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            # Set up raw configs with an unsafe field change to trigger full reload
            runtime._raw_config = {"transition_rules": []}
            runtime._read_raw_config = Mock(
                return_value={"transition_rules": [{"from": "a", "to": "b"}]}
            )

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._run_cortex_loop = AsyncMock()

            await runtime._reload_config()

            mock_load_config.assert_called_once_with(
                "test_config", mode_source_path="/fake/path/test_config.json5"
            )
            runtime._stop_current_orchestrators.assert_called_once()
            runtime._initialize_mode.assert_called_once_with("test_mode")
            runtime._start_orchestrators.assert_called_once()

            assert runtime.mode_config == new_mock_config
            assert runtime.mode_manager.config == new_mock_config

    @pytest.mark.asyncio
    async def test_reload_config_mode_not_found(self, mock_system_config):
        """Test config reload when current mode is not in new config."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
            patch("runtime.multi_mode.cortex.load_mode_config") as mock_load_config,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "old_mode"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.default_mode = "default_mode"
            new_mock_config.modes = {"default_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            # Set up raw configs with an unsafe change to trigger full reload
            runtime._raw_config = {"name": "old"}
            runtime._read_raw_config = Mock(return_value={"name": "new"})

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._run_cortex_loop = AsyncMock()

            await runtime._reload_config()

            runtime._initialize_mode.assert_called_once_with("default_mode")
            assert runtime.mode_manager.state.current_mode == "default_mode"

    @pytest.mark.asyncio
    async def test_reload_config_failure(self, mock_system_config):
        """Test config reload failure handling."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
            patch(
                "runtime.multi_mode.cortex.load_mode_config",
                side_effect=Exception("Load failed"),
            ),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            # Set up raw configs with an unsafe change to trigger full reload
            runtime._raw_config = {"name": "old"}
            runtime._read_raw_config = Mock(return_value={"name": "new"})

            runtime._stop_current_orchestrators = AsyncMock()

            await runtime._reload_config()

            runtime._stop_current_orchestrators.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_hot_reload_enabled(self, mock_system_config):
        """Test run method with hot reload enabled."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(
                return_value=True
            )
            mock_system_config.modes = {"test_mode": Mock()}
            mock_system_config.modes["test_mode"].execute_lifecycle_hooks = AsyncMock()

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True, check_interval=1
            )
            runtime.mode_manager = mock_manager

            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()
            runtime._check_config_changes = AsyncMock()

            call_count = 0
            original_gather = asyncio.gather

            async def mock_gather_with_exit(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    await asyncio.sleep(0.01)
                    raise KeyboardInterrupt()
                return await original_gather(*args, **kwargs)

            with patch("asyncio.gather", side_effect=mock_gather_with_exit):
                try:
                    await runtime.run()
                except KeyboardInterrupt:
                    pass

            assert runtime.config_watcher_task is not None

            runtime._initialize_mode.assert_called_once_with("test_mode")
            runtime._start_orchestrators.assert_called_once()
            runtime._cleanup_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_tasks_with_config_watcher(self, mock_system_config):
        """Test cleanup includes config watcher task when hot reload is enabled."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/test_config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            mock_config_watcher = Mock()
            mock_config_watcher.done.return_value = False
            mock_config_watcher.cancel = Mock()
            runtime.config_watcher_task = mock_config_watcher

            with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
                await runtime._cleanup_tasks()

                mock_config_watcher.cancel.assert_called_once()
                mock_gather.assert_called_once()


BASE_MODE_RAW_CONFIG = {
    "version": "v1.0.2",
    "name": "test_system",
    "default_mode": "patrol",
    "system_governance": "Be safe.",
    "system_prompt_examples": "Example: Hello!",
    "modes": {
        "patrol": {
            "system_prompt_base": "You are patrolling.",
            "hertz": 1.0,
            "agent_inputs": [],
            "cortex_llm": {"type": "OpenAILLM"},
            "agent_actions": [],
        },
    },
    "transition_rules": [],
}


class TestModeHotReloadSafeFields:
    """Tests for MODE_HOT_RELOAD_SAFE_FIELDS and MODE_LEVEL_SAFE_FIELDS constants."""

    def test_global_safe_fields(self):
        """Verify global-level safe fields."""
        assert "system_governance" in MODE_HOT_RELOAD_SAFE_FIELDS
        assert "system_prompt_examples" in MODE_HOT_RELOAD_SAFE_FIELDS

    def test_global_unsafe_fields(self):
        """Verify unsafe fields are not in global safe fields."""
        assert "modes" not in MODE_HOT_RELOAD_SAFE_FIELDS
        assert "transition_rules" not in MODE_HOT_RELOAD_SAFE_FIELDS
        assert "default_mode" not in MODE_HOT_RELOAD_SAFE_FIELDS

    def test_mode_level_safe_fields(self):
        """Verify mode-level safe fields."""
        assert "system_prompt_base" in MODE_LEVEL_SAFE_FIELDS
        assert "hertz" in MODE_LEVEL_SAFE_FIELDS

    def test_mode_level_unsafe_fields(self):
        """Verify unsafe fields are not in mode-level safe fields."""
        assert "agent_inputs" not in MODE_LEVEL_SAFE_FIELDS
        assert "agent_actions" not in MODE_LEVEL_SAFE_FIELDS
        assert "cortex_llm" not in MODE_LEVEL_SAFE_FIELDS


class TestModeDetectConfigChanges:
    """Tests for _detect_config_changes in ModeCortexRuntime."""

    @pytest.fixture
    def mode_runtime(self, mock_system_config):
        """Create a ModeCortexRuntime for testing."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=False
            )
            return runtime

    def test_no_changes(self, mode_runtime):
        """Test identical configs return empty set."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {**BASE_MODE_RAW_CONFIG}
        assert mode_runtime._detect_config_changes(old, new) == set()

    def test_detects_global_governance_change(self, mode_runtime):
        """Test detection of system_governance change."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {**BASE_MODE_RAW_CONFIG, "system_governance": "New rules."}
        changed = mode_runtime._detect_config_changes(old, new)
        assert "system_governance" in changed

    def test_detects_modes_change(self, mode_runtime):
        """Test detection of modes structure change."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "agent_inputs": [{"type": "NewInput"}],
                },
            },
        }
        changed = mode_runtime._detect_config_changes(old, new)
        assert "modes" in changed


class TestModeAreModeChangesSafe:
    """Tests for _are_mode_changes_safe method."""

    @pytest.fixture
    def mode_runtime(self, mock_system_config):
        """Create a ModeCortexRuntime for testing."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=False
            )
            return runtime

    def test_safe_mode_prompt_change(self, mode_runtime):
        """Test that system_prompt_base change within a mode is safe."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "system_prompt_base": "New patrol prompt.",
                },
            },
        }
        assert mode_runtime._are_mode_changes_safe(old, new) is True

    def test_safe_mode_hertz_change(self, mode_runtime):
        """Test that hertz change within a mode is safe."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "hertz": 5.0,
                },
            },
        }
        assert mode_runtime._are_mode_changes_safe(old, new) is True

    def test_unsafe_mode_inputs_change(self, mode_runtime):
        """Test that agent_inputs change within a mode is not safe."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "agent_inputs": [{"type": "NewInput"}],
                },
            },
        }
        assert mode_runtime._are_mode_changes_safe(old, new) is False

    def test_unsafe_mode_added(self, mode_runtime):
        """Test that adding a new mode is not safe."""
        old = {**BASE_MODE_RAW_CONFIG}
        new = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                **BASE_MODE_RAW_CONFIG["modes"],
                "greet": {
                    "system_prompt_base": "Greet people.",
                    "hertz": 1.0,
                },
            },
        }
        assert mode_runtime._are_mode_changes_safe(old, new) is False


class TestModeSelectiveReload:
    """Tests for selective _reload_config in multi-mode."""

    @pytest.fixture
    def mode_runtime(self, mock_system_config):
        """Create a ModeCortexRuntime for selective reload testing."""
        with (
            patch("runtime.multi_mode.cortex.ModeManager") as mock_manager_class,
            patch("runtime.multi_mode.cortex.IOProvider"),
            patch("runtime.multi_mode.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "patrol"
            mock_manager._get_runtime_config_path = Mock(
                return_value="/fake/path/config.json5"
            )
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime._raw_config = {**BASE_MODE_RAW_CONFIG}
            runtime._full_reload = AsyncMock()  # type: ignore[method-assign]

            # Mock mode_config with accessible modes
            mock_mode_obj = Mock()
            mock_mode_obj.system_prompt_base = "You are patrolling."
            mock_mode_obj.hertz = 1.0
            runtime.mode_config = Mock()
            runtime.mode_config.system_governance = "Be safe."
            runtime.mode_config.system_prompt_examples = "Example: Hello!"
            runtime.mode_config.modes = {"patrol": mock_mode_obj}

            runtime.current_config = Mock()

            return runtime

    @pytest.mark.asyncio
    async def test_safe_global_change_no_restart(self, mode_runtime):
        """Test that global safe field change doesn't trigger full reload."""
        new_raw = {**BASE_MODE_RAW_CONFIG, "system_governance": "New rules."}

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        mode_runtime._full_reload.assert_not_called()
        assert mode_runtime.mode_config.system_governance == "New rules."

    @pytest.mark.asyncio
    async def test_safe_mode_prompt_change_no_restart(self, mode_runtime):
        """Test that mode-level prompt change doesn't trigger full reload."""
        new_raw = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "system_prompt_base": "New patrol prompt.",
                },
            },
        }

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        mode_runtime._full_reload.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsafe_change_triggers_full_reload(self, mode_runtime):
        """Test that unsafe field change triggers full reload."""
        new_raw = {
            **BASE_MODE_RAW_CONFIG,
            "transition_rules": [{"from": "patrol", "to": "greet"}],
        }

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        mode_runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsafe_mode_inputs_triggers_full_reload(self, mode_runtime):
        """Test that mode-level unsafe change triggers full reload."""
        new_raw = {
            **BASE_MODE_RAW_CONFIG,
            "modes": {
                "patrol": {
                    **BASE_MODE_RAW_CONFIG["modes"]["patrol"],
                    "agent_inputs": [{"type": "NewInput"}],
                },
            },
        }

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        mode_runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_changes_does_nothing(self, mode_runtime):
        """Test that no changes result in no action."""
        new_raw = {**BASE_MODE_RAW_CONFIG}

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        mode_runtime._full_reload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_config_updated_after_safe_reload(self, mode_runtime):
        """Test that _raw_config is updated after safe change."""
        new_raw = {**BASE_MODE_RAW_CONFIG, "system_governance": "Updated."}

        with patch.object(mode_runtime, "_read_raw_config", return_value=new_raw):
            await mode_runtime._reload_config()

        assert mode_runtime._raw_config["system_governance"] == "Updated."
