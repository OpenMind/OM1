import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from runtime.config import ModeConfig, ModeSystemConfig, RuntimeConfig
from runtime.cortex import ModeCortexRuntime
from runtime.hot_reload import ConfigChange, HotReloadManager, ReloadStrategy


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
    manager = Mock()
    manager.current_mode_name = "default"
    manager.add_transition_callback = Mock()
    manager.process_tick = AsyncMock(return_value=None)
    # Hot-reload path property
    manager.runtime_config_path = "/fake/path/test_config.json5"
    return manager


@pytest.fixture
def mock_orchestrators():
    return {
        "fuser": Mock(),
        "action_orchestrator": Mock(),
        "simulator_orchestrator": Mock(),
        "background_orchestrator": Mock(),
        "input_orchestrator": Mock(),
    }


@pytest.fixture
def cortex_runtime(mock_system_config):
    """ModeCortexRuntime instance for testing (without hot-reload)."""
    with (
        patch("runtime.cortex.ModeManager") as mock_manager_class,
        patch("runtime.cortex.IOProvider") as mock_io_provider_class,
        patch("runtime.cortex.SleepTickerProvider") as mock_sleep_provider_class,
    ):
        mock_manager = Mock()
        mock_manager.current_mode_name = "default"
        mock_manager.add_transition_callback = Mock()
        mock_manager.runtime_config_path = "/fake/path/test_config.json5"
        mock_manager_class.return_value = mock_manager

        mock_io_provider = Mock()
        mock_io_provider_class.return_value = mock_io_provider

        mock_sleep_provider = Mock()
        mock_sleep_provider.skip_sleep = False
        mock_sleep_provider_class.return_value = mock_sleep_provider

        runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)
        runtime.mode_manager = mock_manager
        runtime.io_provider = mock_io_provider
        runtime.sleep_ticker_provider = mock_sleep_provider

        return runtime, {
            "mode_manager": mock_manager,
            "io_provider": mock_io_provider,
            "sleep_provider": mock_sleep_provider,
        }


class TestModeCortexRuntime:
    """Test cases for ModeCortexRuntime – non-hot-reload functionality."""

    def test_initialization(self, mock_system_config):
        """Test basic initialization without hot-reload."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            # Explicitly disable hot-reload for this test
            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=False
            )

            assert runtime.mode_config == mock_system_config
            assert runtime.mode_config_name == "test_config"
            assert runtime.current_config is None
            assert runtime.fuser is None
            assert runtime.action_orchestrator is None
            assert runtime.simulator_orchestrator is None
            assert runtime.background_orchestrator is None
            assert runtime.input_orchestrator is None
            assert runtime._mode_initialized is False
            assert runtime.config_watcher is None

            mock_manager.add_transition_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mode(self, cortex_runtime, mock_mode_config):
        runtime, mocks = cortex_runtime

        with (
            patch("runtime.cortex.Fuser") as mock_fuser_class,
            patch("runtime.cortex.ActionOrchestrator") as mock_action_class,
            patch("runtime.cortex.SimulatorOrchestrator") as mock_simulator_class,
            patch("runtime.cortex.BackgroundOrchestrator") as mock_background_class,
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
        runtime, mocks = cortex_runtime

        with (
            patch.object(runtime, "_stop_current_orchestrators") as mock_stop,
            patch.object(runtime, "_initialize_mode") as mock_init,
            patch.object(runtime, "_start_orchestrators") as mock_start,
        ):
            runtime.mode_config.modes = {
                "from_mode": Mock(),
                "to_mode": Mock(),
            }

            await runtime._on_mode_transition("from_mode", "to_mode")

            mock_stop.assert_called_once()
            mock_init.assert_called_once_with("to_mode")
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_mode_transition_exception(self, cortex_runtime):
        runtime, mocks = cortex_runtime

        runtime.mode_config.modes = {
            "from_mode": Mock(),
            "to_mode": Mock(),
        }

        with patch.object(
            runtime, "_stop_current_orchestrators", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception, match="Test error"):
                await runtime._on_mode_transition("from_mode", "to_mode")

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators(self, cortex_runtime):
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
        runtime, mocks = cortex_runtime
        runtime.current_config = None

        with pytest.raises(RuntimeError, match="No current config available"):
            await runtime._start_orchestrators()

    @pytest.mark.asyncio
    async def test_cleanup_tasks(self, cortex_runtime):
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
    """Test cases for hot-reload functionality (watchdog-based)."""

    def test_hot_reload_initialization_enabled(self, mock_system_config):
        """Test hot-reload initialization when enabled – ConfigFileWatcher created."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.ConfigFileWatcher") as mock_watcher_class,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )

            assert runtime.hot_reload is True
            assert runtime.config_watcher is not None
            mock_watcher_class.assert_called_once_with(
                config_path="/fake/path/test_config.json5",
                on_change_callback=runtime._reload_config,
                debounce_seconds=0.5,
            )

    def test_hot_reload_initialization_disabled(self, mock_system_config):
        """Test hot-reload initialization when disabled – no watcher."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.ConfigFileWatcher") as mock_watcher_class,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=False
            )

            assert runtime.hot_reload is False
            assert runtime.config_watcher is None
            mock_watcher_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_config_no_changes_detected(self, mock_system_config, caplog):
        """Test reload when no registered fields changed – returns early."""
        import logging

        mock_system_config.modes = {"test_mode": Mock()}

        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
            patch("runtime.config.mode_config_to_dict", return_value={}),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime.hot_reload_manager = Mock(spec=HotReloadManager)
            runtime.hot_reload_manager.detect_changes = Mock(return_value=[])
            runtime._stop_current_orchestrators = AsyncMock()

            with caplog.at_level(logging.INFO):
                await runtime._reload_config()

            assert "no registered fields modified" in caplog.text.lower()
            runtime._stop_current_orchestrators.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_config_selective_hot_reload(self, mock_system_config):
        """Test selective hot-reload for system_prompt change – no restart."""

        mock_system_config.modes = {"test_mode": Mock()}

        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
            patch("runtime.config.mode_config_to_dict") as mock_to_dict,
            patch("runtime.cortex.Fuser") as mock_fuser_class,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            mock_to_dict.return_value = {}

            change = ConfigChange(
                field_path="system_prompt_base",
                old_value="old",
                new_value="new",
                strategy=ReloadStrategy.HOT_RELOAD,
            )

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime.hot_reload_manager = Mock(spec=HotReloadManager)
            runtime.hot_reload_manager.detect_changes = Mock(return_value=[change])
            runtime.hot_reload_manager.categorize_changes = Mock(
                return_value={
                    ReloadStrategy.RESTART_REQUIRED: [],
                    ReloadStrategy.VALIDATE_FIRST: [],
                    ReloadStrategy.HOT_RELOAD: [change],
                }
            )
            runtime.hot_reload_manager.track_change = Mock()

            mock_runtime_config = Mock(spec=RuntimeConfig)
            mock_mode_entry = new_mock_config.modes["test_mode"]
            mock_mode_entry.to_runtime_config = Mock(return_value=mock_runtime_config)

            await runtime._reload_config()

            mock_fuser_class.assert_called_once_with(mock_runtime_config)
            runtime.hot_reload_manager.track_change.assert_called_once_with(change)

    @pytest.mark.asyncio
    async def test_reload_config_validate_first_success(self, mock_system_config):
        """Test selective hot-reload with VALIDATE_FIRST – validation passes."""

        mock_system_config.modes = {"test_mode": Mock()}

        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
            patch("runtime.config.mode_config_to_dict", return_value={}),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            change = ConfigChange(
                field_path="cortex_llm.config.temperature",
                old_value=0.7,
                new_value=0.9,
                strategy=ReloadStrategy.VALIDATE_FIRST,
            )

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime.hot_reload_manager = Mock(spec=HotReloadManager)
            runtime.hot_reload_manager.detect_changes = Mock(return_value=[change])
            runtime.hot_reload_manager.categorize_changes = Mock(
                return_value={
                    ReloadStrategy.RESTART_REQUIRED: [],
                    ReloadStrategy.VALIDATE_FIRST: [change],
                    ReloadStrategy.HOT_RELOAD: [],
                }
            )
            runtime.hot_reload_manager.validate_changes = Mock(
                return_value={"cortex_llm.config.temperature": True}
            )
            runtime.hot_reload_manager.track_change = Mock()

            mock_runtime_config = Mock(spec=RuntimeConfig)
            mock_mode_entry = new_mock_config.modes["test_mode"]
            mock_mode_entry.to_runtime_config = Mock(return_value=mock_runtime_config)

            mock_action_orch = Mock()
            mock_action_orch.update_config = Mock()
            runtime.action_orchestrator = mock_action_orch

            await runtime._reload_config()

            mock_action_orch.update_config.assert_called_once_with(mock_runtime_config)
            runtime.hot_reload_manager.track_change.assert_called_once_with(change)

    @pytest.mark.asyncio
    async def test_reload_config_validate_first_failure(
        self, mock_system_config, caplog
    ):
        """Test selective hot-reload aborts when validation fails."""
        import logging

        mock_system_config.modes = {"test_mode": Mock()}

        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
            patch("runtime.config.mode_config_to_dict", return_value={}),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            change = ConfigChange(
                field_path="cortex_llm.config.temperature",
                old_value=0.7,
                new_value=5.0,
                strategy=ReloadStrategy.VALIDATE_FIRST,
            )

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime.hot_reload_manager = Mock(spec=HotReloadManager)
            runtime.hot_reload_manager.detect_changes = Mock(return_value=[change])
            runtime.hot_reload_manager.categorize_changes = Mock(
                return_value={
                    ReloadStrategy.RESTART_REQUIRED: [],
                    ReloadStrategy.VALIDATE_FIRST: [change],
                    ReloadStrategy.HOT_RELOAD: [],
                }
            )
            runtime.hot_reload_manager.validate_changes = Mock(
                return_value={"cortex_llm.config.temperature": False}
            )
            runtime._stop_current_orchestrators = AsyncMock()

            with caplog.at_level(logging.ERROR):
                await runtime._reload_config()

            assert "validation failed" in caplog.text.lower()
            runtime._stop_current_orchestrators.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_config_restart_required(self, mock_system_config, caplog):
        """Test reload triggers full restart when RESTART_REQUIRED changes."""
        import logging

        mock_system_config.modes = {"test_mode": Mock()}

        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
            patch("runtime.config.mode_config_to_dict", return_value={}),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.default_mode = "test_mode"
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            change = ConfigChange(
                field_path="hertz",
                old_value=10,
                new_value=20,
                strategy=ReloadStrategy.RESTART_REQUIRED,
            )

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime.hot_reload_manager = Mock(spec=HotReloadManager)
            runtime.hot_reload_manager.detect_changes = Mock(return_value=[change])
            runtime.hot_reload_manager.categorize_changes = Mock(
                return_value={
                    ReloadStrategy.RESTART_REQUIRED: [change],
                    ReloadStrategy.VALIDATE_FIRST: [],
                    ReloadStrategy.HOT_RELOAD: [],
                }
            )
            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()

            with caplog.at_level(logging.WARNING):
                await runtime._reload_config()

            assert "restart required" in caplog.text.lower()
            runtime._stop_current_orchestrators.assert_called_once()
            runtime._initialize_mode.assert_called_once()
            runtime._start_orchestrators.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_hot_reload_enabled(self, mock_system_config):
        """Test run() starts the config watcher."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.ConfigFileWatcher") as mock_watcher_class,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(
                return_value=True
            )
            mock_system_config.modes = {"test_mode": Mock()}
            mock_system_config.modes["test_mode"].execute_lifecycle_hooks = AsyncMock()

            mock_watcher = Mock()
            mock_watcher.start = Mock()
            mock_watcher_class.return_value = mock_watcher

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

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

            mock_watcher.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_tasks_with_config_watcher(self, mock_system_config):
        """Test cleanup stops the config watcher."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            mock_watcher = Mock()
            mock_watcher.stop = Mock()
            runtime.config_watcher = mock_watcher

            with patch("asyncio.gather", new_callable=AsyncMock):
                await runtime._cleanup_tasks()

            mock_watcher.stop.assert_called_once()
            assert runtime.config_watcher is None


class TestHotReloadMultiToSingle:
    """Test hot reload when switching from multi-mode to single-mode config."""

    @pytest.mark.asyncio
    async def test_reload_multi_to_single_mode(self, mock_system_config):
        """Test full restart when current mode is missing in new config."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "mode_1"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager.runtime_config_path = "/fake/path/test_config.json5"
            mock_manager_class.return_value = mock_manager

            mock_system_config.modes = {
                "mode_1": Mock(),
                "mode_2": Mock(),
            }
            mock_system_config.default_mode = "mode_1"

            single_mode_mock = Mock(spec=ModeConfig)
            single_mode_mock.name = "single_mode"
            single_mode_mock.display_name = "single_mode"

            new_single_config = Mock(spec=ModeSystemConfig)
            new_single_config.default_mode = "single_mode"
            new_single_config.modes = {"single_mode": single_mode_mock}
            mock_load_config.return_value = new_single_config

            runtime = ModeCortexRuntime(
                mock_system_config, "test_config", hot_reload=True
            )
            runtime.mode_manager = mock_manager

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()

            await runtime._reload_config()

            runtime._initialize_mode.assert_called_once_with("single_mode")
            assert runtime.mode_manager.state.current_mode == "single_mode"
            assert runtime.mode_config == new_single_config
            assert runtime.mode_manager.config == new_single_config

            runtime._stop_current_orchestrators.assert_called_once()
            runtime._start_orchestrators.assert_called_once()

            assert len(runtime.mode_manager.state.transition_history) == 1
            assert (
                "config_reload->single_mode:full_restart"
                in runtime.mode_manager.state.transition_history[0]
            )
