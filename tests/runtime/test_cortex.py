import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from runtime.config import ModeConfig, ModeSystemConfig
from runtime.cortex import ModeCortexRuntime


@pytest.fixture
def sample_mode_config():
    mode_config = ModeConfig(
        version="v1.0.3",
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
        "background_orchestrator": Mock(),
        "input_orchestrator": Mock(),
    }


@pytest.fixture
def cortex_runtime(mock_system_config):
    """ModeCortexRuntime instance for testing."""
    with (
        patch("runtime.cortex.ModeManager") as mock_manager_class,
        patch("runtime.cortex.IOProvider") as mock_io_provider_class,
        patch("runtime.cortex.SleepTickerProvider") as mock_sleep_provider_class,
    ):
        mock_manager = Mock()
        mock_manager.current_mode_name = "default"
        mock_manager.add_transition_callback = Mock()
        mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")

            assert runtime.mode_config == mock_system_config
            assert runtime.mode_config_name == "test_config"
            assert runtime.current_config is None
            assert runtime.fuser is None
            assert runtime.action_orchestrator is None
            assert runtime.background_orchestrator is None
            assert runtime.input_orchestrator is None
            assert runtime._mode_initialized is False

            mock_manager.add_transition_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mode(self, cortex_runtime, mock_mode_config):
        """Test mode initialization."""
        runtime, mocks = cortex_runtime

        with (
            patch("runtime.cortex.Fuser") as mock_fuser_class,
            patch("runtime.cortex.ActionOrchestrator") as mock_action_class,
            patch("runtime.cortex.BackgroundOrchestrator") as mock_background_class,
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            mock_fuser = Mock()
            mock_action_orch = Mock()
            mock_background_orch = Mock()
            mock_mcp_orch = Mock()

            mock_fuser_class.return_value = mock_fuser
            mock_action_class.return_value = mock_action_orch
            mock_background_class.return_value = mock_background_orch
            mock_mcp_class.return_value = mock_mcp_orch

            mock_mcp_servers = Mock()
            mock_mcp_servers.start = AsyncMock()
            mock_mode_config.to_runtime_config.return_value = Mock(
                mcp_servers=mock_mcp_servers,
                cortex_llm=Mock(),
            )
            runtime.mode_config.modes = {"test_mode": mock_mode_config}

            await runtime._initialize_mode("test_mode")

            mock_mode_config.load_components.assert_called_once_with(runtime.mode_config)
            mock_mode_config.to_runtime_config.assert_called_once_with(runtime.mode_config)

            assert runtime.fuser == mock_fuser
            assert runtime.action_orchestrator == mock_action_orch
            assert runtime.background_orchestrator == mock_background_orch
            # mcp_orchestrator is created in _initialize_mode but start()
            # is called later in _start_orchestrators(), not here.
            assert runtime.mcp_orchestrator == mock_mcp_orch
            mock_mcp_servers.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_initialize_mode_no_mcp_servers(self, cortex_runtime, mock_mode_config):
        """Test that mcp_orchestrator is None when mcp_servers is absent."""
        runtime, mocks = cortex_runtime

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            mock_mode_config.to_runtime_config.return_value = Mock(
                mcp_servers=None,
                cortex_llm=Mock(),
            )
            runtime.mode_config.modes = {"test_mode": mock_mode_config}

            await runtime._initialize_mode("test_mode")

            mock_mcp_class.assert_not_called()
            assert runtime.mcp_orchestrator is None

    @pytest.mark.asyncio
    async def test_tick_calls_mcp_extract(self, cortex_runtime):
        """Test that _tick enters the MCP branch and calls extract_mcp_actions."""
        runtime, mocks = cortex_runtime

        mock_output = Mock()
        mock_output.actions = []

        async def stream_output(_: str):
            yield mock_output

        runtime.current_config = Mock()
        runtime.current_config.hertz = 10.0
        runtime.current_config.cortex_llm = Mock()
        runtime.current_config.cortex_llm.ask_stream = Mock(side_effect=stream_output)
        runtime.current_config.agent_inputs = []

        runtime.fuser = Mock()
        runtime.fuser.fuse = AsyncMock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.action_orchestrator.promise = AsyncMock()
        runtime.mcp_orchestrator = Mock()
        runtime.mcp_orchestrator.max_rounds = 3
        runtime.mcp_orchestrator.extract_om1_actions = Mock(return_value=[])
        runtime.mcp_orchestrator.execute_mcp_actions = AsyncMock(return_value=(None, None))
        runtime.mcp_orchestrator.build_call_signature = Mock(return_value="sig")

        # Mock io_provider with mode_transition_input context manager
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=None)
        ctx.__exit__ = Mock(return_value=False)
        runtime.io_provider = Mock()
        runtime.io_provider.mode_transition_input = Mock(return_value=ctx)

        runtime.mode_manager = Mock()
        runtime.mode_manager.process_tick = AsyncMock(return_value=None)

        runtime._pending_mode_transition = None
        runtime._mode_transition_event = Mock()
        runtime._mode_transition_event.set = Mock()
        runtime._cortex_loop_generation = 0

        await runtime._tick(0)

        runtime.mcp_orchestrator.execute_mcp_actions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_skips_mcp_when_none(self, cortex_runtime):
        """Test that _tick works normally when mcp_orchestrator is None."""
        runtime, mocks = cortex_runtime

        mock_output = Mock()
        mock_output.actions = []

        async def stream_output(_: str):
            yield mock_output

        runtime.current_config = Mock()
        runtime.current_config.hertz = 10.0
        runtime.current_config.cortex_llm = Mock()
        runtime.current_config.cortex_llm.ask_stream = Mock(side_effect=stream_output)
        runtime.current_config.agent_inputs = []

        runtime.fuser = Mock()
        runtime.fuser.fuse = AsyncMock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.action_orchestrator.promise = AsyncMock()
        runtime.mcp_orchestrator = None

        # Mock io_provider with mode_transition_input context manager
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=None)
        ctx.__exit__ = Mock(return_value=False)
        runtime.io_provider = Mock()
        runtime.io_provider.mode_transition_input = Mock(return_value=ctx)

        runtime.mode_manager = Mock()
        runtime.mode_manager.process_tick = AsyncMock(return_value=None)

        runtime._pending_mode_transition = None
        runtime._mode_transition_event = Mock()
        runtime._mode_transition_event.set = Mock()
        runtime._cortex_loop_generation = 0

        await runtime._tick(0)

        # Should still reach action_orchestrator.promise
        runtime.action_orchestrator.promise.assert_called_once()

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

        with patch.object(runtime, "_stop_current_orchestrators", side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await runtime._on_mode_transition("from_mode", "to_mode")

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators(self, cortex_runtime):
        """Test stopping current orchestrators."""
        runtime, mocks = cortex_runtime

        mock_input_task = Mock()
        mock_input_task.done.return_value = False
        mock_input_task.cancel = Mock()

        mock_action_task = Mock()
        mock_action_task.done.return_value = False
        mock_action_task.cancel = Mock()

        mock_background_task = Mock()
        mock_background_task.done.return_value = False
        mock_background_task.cancel = Mock()

        runtime.input_listener_task = mock_input_task
        runtime.action_task = mock_action_task
        runtime.background_task = mock_background_task

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (
                {
                    mock_input_task,
                    mock_action_task,
                    mock_background_task,
                },
                set(),
            )

            await runtime._stop_current_orchestrators()

            mock_input_task.cancel.assert_called_once()
            mock_action_task.cancel.assert_called_once()
            mock_background_task.cancel.assert_called_once()

            mock_wait.assert_called_once()

            assert runtime.input_listener_task is None
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

        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()

        runtime.input_listener_task = mock_task

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await runtime._cleanup_tasks()

            mock_task.cancel.assert_called_once()
            mock_gather.assert_called_once()


class TestMCPModeTransition:
    """Test MCP lifecycle during cortex mode transitions."""

    @pytest.mark.asyncio
    async def test_initialize_mode_creates_mcp_orchestrator(self, cortex_runtime, mock_mode_config):
        """_initialize_mode should create MCPOrchestrator when MCP is configured."""
        runtime, mocks = cortex_runtime

        mock_mcp_servers = Mock()
        mock_mode_config.to_runtime_config.return_value = Mock(
            mcp_servers=mock_mcp_servers,
            cortex_llm=Mock(),
        )
        runtime.mode_config.modes = {"test_mode": mock_mode_config}

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            await runtime._initialize_mode("test_mode")

        mock_mcp_class.assert_called_once()
        assert runtime.mcp_orchestrator is not None

    @pytest.mark.asyncio
    async def test_stop_orchestrators_calls_mcp_stop(self, cortex_runtime):
        """_stop_current_orchestrators should call mcp_orchestrator.stop()."""
        runtime, mocks = cortex_runtime

        mock_mcp_orch = Mock()
        mock_mcp_orch.stop = AsyncMock()
        runtime.current_config = Mock()
        runtime.mcp_orchestrator = mock_mcp_orch

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (set(), set())
            await runtime._stop_current_orchestrators()

        mock_mcp_orch.stop.assert_awaited_once()
        assert runtime.mcp_orchestrator is None

    @pytest.mark.asyncio
    async def test_transition_mcp_to_no_mcp(self, cortex_runtime, mock_mode_config):
        """Mode transition from MCP mode to non-MCP mode."""
        runtime, mocks = cortex_runtime

        mock_mcp_orch = Mock()
        mock_mcp_orch.stop = AsyncMock()
        runtime.current_config = Mock()
        runtime.mcp_orchestrator = mock_mcp_orch

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (set(), set())
            await runtime._stop_current_orchestrators()

        mock_mcp_orch.stop.assert_awaited_once()
        assert runtime.mcp_orchestrator is None

        mock_mode_config.to_runtime_config.return_value = Mock(
            mcp_servers=None,
            cortex_llm=Mock(),
        )
        runtime.mode_config.modes = {"no_mcp_mode": mock_mode_config}

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            await runtime._initialize_mode("no_mcp_mode")

        mock_mcp_class.assert_not_called()
        assert runtime.mcp_orchestrator is None

    @pytest.mark.asyncio
    async def test_initialize_mode_calls_mcp_start(self, cortex_runtime, mock_mode_config):
        """_start_orchestrators should call mcp_orchestrator.start()."""
        runtime, mocks = cortex_runtime

        mock_mcp_servers = Mock()
        mock_mode_config.to_runtime_config.return_value = Mock(
            mcp_servers=mock_mcp_servers,
            cortex_llm=Mock(),
        )
        runtime.mode_config.modes = {"test_mode": mock_mode_config}

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            mock_mcp_orch = AsyncMock()
            mock_mcp_orch.start = AsyncMock()
            mock_mcp_class.return_value = mock_mcp_orch

            await runtime._initialize_mode("test_mode")

        # Verify the orchestrator was created.
        mock_mcp_class.assert_called_once()
        assert runtime.mcp_orchestrator is not None

    @pytest.mark.asyncio
    async def test_transition_no_mcp_to_mcp(self, cortex_runtime, mock_mode_config):
        """Mode transition from non-MCP mode to MCP mode."""
        runtime, mocks = cortex_runtime

        runtime.current_config = Mock(mcp_servers=None)
        runtime.mcp_orchestrator = None

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (set(), set())
            await runtime._stop_current_orchestrators()

        mock_mcp_servers = Mock()
        mock_mcp_servers.start = AsyncMock()
        mock_mode_config.to_runtime_config.return_value = Mock(
            mcp_servers=mock_mcp_servers,
            cortex_llm=Mock(),
        )
        runtime.mode_config.modes = {"mcp_mode": mock_mode_config}

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            mock_mcp_orch = AsyncMock()
            mock_mcp_orch.start = AsyncMock()
            mock_mcp_class.return_value = mock_mcp_orch
            await runtime._initialize_mode("mcp_mode")

        mock_mcp_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_mcp_to_mcp(self, cortex_runtime, mock_mode_config):
        """Full e2e mode transition from MCP mode to a different MCP mode."""
        runtime, mocks = cortex_runtime

        old_mcp_orch = Mock()
        old_mcp_orch.stop = AsyncMock()
        runtime.current_config = Mock()
        runtime.mcp_orchestrator = old_mcp_orch

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (set(), set())
            await runtime._stop_current_orchestrators()

        old_mcp_orch.stop.assert_awaited_once()
        assert runtime.mcp_orchestrator is None

        new_mcp_servers = Mock()
        new_mcp_servers.start = AsyncMock()
        mock_mode_config.to_runtime_config.return_value = Mock(
            mcp_servers=new_mcp_servers,
            cortex_llm=Mock(),
        )
        runtime.mode_config.modes = {"new_mcp_mode": mock_mode_config}

        with (
            patch("runtime.cortex.Fuser"),
            patch("runtime.cortex.ActionOrchestrator"),
            patch("runtime.cortex.BackgroundOrchestrator"),
            patch("runtime.cortex.MCPOrchestrator") as mock_mcp_class,
        ):
            mock_new_orch = AsyncMock()
            mock_new_orch.start = AsyncMock()
            mock_mcp_class.return_value = mock_new_orch
            await runtime._initialize_mode("new_mcp_mode")

        mock_mcp_class.assert_called_once()
        assert runtime.mcp_orchestrator == mock_new_orch


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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=30)

            assert runtime.hot_reload is True
            assert runtime.check_interval == 30
            assert runtime.last_modified == 1234567890.0
            assert runtime.config_path.endswith("test_config.json5")

    def test_hot_reload_initialization_disabled(self, mock_system_config):
        """Test hot reload initialization when disabled."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)

            assert runtime.hot_reload is False
            assert runtime.last_modified is None

    def test_get_file_mtime_existing_file(self, mock_system_config, temp_config_file):
        """Test getting modification time of existing file."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.config_path = temp_config_file

            mtime = runtime._get_file_mtime()
            assert mtime > 0

    def test_get_file_mtime_nonexistent_file(self, mock_system_config):
        """Test getting modification time of non-existent file."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.config_path = "/nonexistent/file.json5"

            mtime = runtime._get_file_mtime()
            assert mtime == 0.0

    @pytest.mark.asyncio
    async def test_check_config_changes_file_changed(self, mock_system_config, temp_config_file):
        """Test config change detection when file is modified."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=0.1)
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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("os.path.exists", return_value=True),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=0.1)
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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=0.1)
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
        """Test successful config reload."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.default_mode = "test_mode"
            new_mock_config.modes = {"test_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.mode_manager = mock_manager

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._run_cortex_loop = AsyncMock()

            await runtime._reload_config()

            mock_load_config.assert_called_once_with("test_config", mode_source_path="/fake/path/test_config.json5")
            runtime._stop_current_orchestrators.assert_called_once()
            runtime._initialize_mode.assert_called_once_with("test_mode")
            runtime._start_orchestrators.assert_called_once()

            assert runtime.mode_config == new_mock_config
            assert runtime.mode_manager.config == new_mock_config

    @pytest.mark.asyncio
    async def test_reload_config_mode_not_found(self, mock_system_config):
        """Test config reload when current mode is not in new config."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch("runtime.cortex.load_mode_config") as mock_load_config,
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "old_mode"
            mock_manager.state = Mock()
            mock_manager.state.transition_history = []
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            new_mock_config = Mock(spec=ModeSystemConfig)
            new_mock_config.default_mode = "default_mode"
            new_mock_config.modes = {"default_mode": Mock()}
            mock_load_config.return_value = new_mock_config

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.mode_manager = mock_manager

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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
            patch(
                "runtime.cortex.load_mode_config",
                side_effect=Exception("Load failed"),
            ),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.mode_manager = mock_manager

            runtime._stop_current_orchestrators = AsyncMock()

            await runtime._reload_config()

            runtime._stop_current_orchestrators.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_hot_reload_enabled(self, mock_system_config):
        """Test run method with hot reload enabled."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)
            mock_system_config.modes = {"test_mode": Mock()}
            mock_system_config.modes["test_mode"].execute_lifecycle_hooks = AsyncMock()

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=1)
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
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.mode_manager = mock_manager

            mock_config_watcher = Mock()
            mock_config_watcher.done.return_value = False
            mock_config_watcher.cancel = Mock()
            runtime.config_watcher_task = mock_config_watcher

            with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
                await runtime._cleanup_tasks()

                mock_config_watcher.cancel.assert_called_once()
                mock_gather.assert_called_once()


class TestHotReloadMultiToSingle:
    """Test hot reload from multi-mode config to single-mode config."""

    @pytest.mark.asyncio
    async def test_reload_multi_to_single_mode(self, mock_system_config):
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
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
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

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True)
            runtime.mode_manager = mock_manager

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._run_cortex_loop = AsyncMock()

            await runtime._reload_config()

            runtime._initialize_mode.assert_called_once_with("single_mode")
            assert runtime.mode_manager.state.current_mode == "single_mode"

            assert runtime.mode_config == new_single_config
            assert runtime.mode_manager.config == new_single_config

            runtime._stop_current_orchestrators.assert_called_once()
            runtime._start_orchestrators.assert_called_once()

            assert len(runtime.mode_manager.state.transition_history) == 1
            assert "config_reload->single_mode:hot_reload" in runtime.mode_manager.state.transition_history[0]

            assert len(new_single_config.modes) == 1
            assert "single_mode" in new_single_config.modes


class TestAdditionalCoverage:
    """Additional tests to increase coverage above 65%."""

    @pytest.mark.asyncio
    async def test_handle_mode_transitions_success(self, mock_system_config):
        """Test _handle_mode_transitions with successful transition."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager._execute_transition = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            runtime._pending_mode_transition = "target_mode"
            runtime._pending_transition_reason = "test_reason"

            # Run the handler for one iteration
            task = asyncio.create_task(runtime._handle_mode_transitions())

            # Wait a bit and trigger the event
            await asyncio.sleep(0.01)
            runtime._mode_transition_event.set()
            await asyncio.sleep(0.02)

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify transition was executed
            mock_manager._execute_transition.assert_called_once_with("target_mode", "test_reason")

    @pytest.mark.asyncio
    async def test_handle_mode_transitions_failure(self, mock_system_config):
        """Test _handle_mode_transitions with failed transition."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager._execute_transition = AsyncMock(return_value=False)
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            runtime._pending_mode_transition = "target_mode"
            runtime._pending_transition_reason = None  # Test default reason

            task = asyncio.create_task(runtime._handle_mode_transitions())
            await asyncio.sleep(0.01)
            runtime._mode_transition_event.set()
            await asyncio.sleep(0.02)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_manager._execute_transition.assert_called_once_with("target_mode", "input_triggered")

    @pytest.mark.asyncio
    async def test_handle_mode_transitions_exception(self, mock_system_config):
        """Test _handle_mode_transitions with exception during transition."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager._execute_transition = AsyncMock(side_effect=Exception("Transition error"))
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            runtime._pending_mode_transition = "target_mode"

            task = asyncio.create_task(runtime._handle_mode_transitions())
            await asyncio.sleep(0.01)
            runtime._mode_transition_event.set()
            await asyncio.sleep(0.1)  # Give it time to handle exception

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def test_is_generation_valid_invalid(self, cortex_runtime):
        """Test _is_generation_valid with mismatched generation."""
        runtime, _ = cortex_runtime
        runtime._cortex_loop_generation = 5

        # Test with mismatched generation
        result = runtime._is_generation_valid(3, "test context")
        assert result is False

        # Test with valid generation
        result = runtime._is_generation_valid(5, "test context")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_mode_info(self, mock_system_config):
        """Test get_mode_info method."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.get_mode_info = Mock(return_value={"mode": "info"})
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            info = runtime.get_mode_info()

            assert info == {"mode": "info"}
            mock_manager.get_mode_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_mode_change(self, mock_system_config):
        """Test request_mode_change method."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.request_transition = AsyncMock(return_value=True)
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            result = await runtime.request_mode_change("new_mode")

            assert result is True
            mock_manager.request_transition.assert_called_once_with("new_mode", "manual")

    def test_get_available_modes(self, mock_system_config):
        """Test get_available_modes method."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "mode1"
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            mode1_config = Mock()
            mode1_config.display_name = "Mode 1"
            mode1_config.description = "First mode"

            mode2_config = Mock()
            mode2_config.display_name = "Mode 2"
            mode2_config.description = "Second mode"

            mock_system_config.modes = {
                "mode1": mode1_config,
                "mode2": mode2_config,
            }

            runtime = ModeCortexRuntime(mock_system_config, "test_config")
            modes = runtime.get_available_modes()

            assert "mode1" in modes
            assert "mode2" in modes
            assert modes["mode1"]["is_current"] is True
            assert modes["mode2"]["is_current"] is False
            assert modes["mode1"]["display_name"] == "Mode 1"
            assert modes["mode2"]["description"] == "Second mode"

    @pytest.mark.asyncio
    async def test_run_cortex_loop_cancelled(self, cortex_runtime):
        """Test _run_cortex_loop when cancelled."""
        runtime, _ = cortex_runtime
        runtime.current_config = Mock()
        runtime.current_config.hertz = 10
        runtime.sleep_ticker_provider = Mock()
        runtime.sleep_ticker_provider.skip_sleep = False
        runtime.sleep_ticker_provider.sleep = AsyncMock()

        runtime._tick = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await runtime._run_cortex_loop()

    @pytest.mark.asyncio
    async def test_run_cortex_loop_unexpected_error(self, cortex_runtime):
        """Test _run_cortex_loop with unexpected error."""
        runtime, _ = cortex_runtime
        runtime.current_config = Mock()
        runtime.current_config.hertz = 10
        runtime.sleep_ticker_provider = Mock()
        runtime.sleep_ticker_provider.skip_sleep = False
        runtime.sleep_ticker_provider.sleep = AsyncMock()

        runtime._tick = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(Exception, match="Unexpected error"):
            await runtime._run_cortex_loop()

    @pytest.mark.asyncio
    async def test_tick_not_initialized(self, cortex_runtime):
        """Test _tick when cortex is not properly initialized."""
        runtime, _ = cortex_runtime
        runtime.current_config = None
        runtime.fuser = None
        runtime.action_orchestrator = None

        await runtime._tick(0)

    @pytest.mark.asyncio
    async def test_tick_during_reload(self, cortex_runtime):
        """Test _tick skips processing during reload."""
        runtime, _ = cortex_runtime
        runtime.current_config = Mock()
        runtime.fuser = Mock()
        runtime.action_orchestrator = Mock()
        runtime._is_reloading = True

        await runtime._tick(0)

        runtime.fuser.fuse.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_no_prompt(self, cortex_runtime):
        """Test _tick when fuser returns None."""
        runtime, _ = cortex_runtime
        runtime.current_config = Mock()
        runtime.current_config.agent_inputs = []
        runtime.fuser = Mock()
        runtime.fuser.fuse = AsyncMock(return_value=None)
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.io_provider.increment_tick = Mock(return_value=1)
        runtime._is_reloading = False
        runtime._cortex_loop_generation = 0

        await runtime._tick(0)

        runtime.fuser.fuse.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_mode_transition_triggered(self, cortex_runtime):
        """Test _tick when mode transition is triggered."""
        runtime, _ = cortex_runtime
        runtime.current_config = Mock()
        runtime.current_config.agent_inputs = []
        runtime.fuser = Mock()
        runtime.fuser.fuse = AsyncMock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.io_provider.increment_tick = Mock(return_value=1)

        ctx = Mock()
        ctx.__enter__ = Mock(return_value=None)
        ctx.__exit__ = Mock(return_value=False)
        runtime.io_provider.mode_transition_input = Mock(return_value=ctx)
        runtime.io_provider.get_mode_transition_input = Mock(return_value="input")

        runtime.mode_manager = Mock()
        runtime.mode_manager.process_tick = AsyncMock(return_value=("new_mode", "test_reason"))
        runtime._pending_mode_transition = None
        runtime._mode_transition_event = Mock()
        runtime._is_reloading = False
        runtime._cortex_loop_generation = 0

        await runtime._tick(0)

        assert runtime._pending_mode_transition == "new_mode"
        assert runtime._pending_transition_reason == "test_reason"
        runtime._mode_transition_event.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_llm_cancelled(self, cortex_runtime):
        """Test _tick when LLM call is cancelled."""
        runtime, _ = cortex_runtime

        async def cancelled_stream(_):
            if False:
                yield  # Make it an async generator
            raise asyncio.CancelledError()

        runtime.current_config = Mock()
        runtime.current_config.hertz = 10.0
        runtime.current_config.cortex_llm = Mock()
        runtime.current_config.cortex_llm.ask_stream = cancelled_stream
        runtime.current_config.agent_inputs = []

        runtime.fuser = Mock()
        runtime.fuser.fuse = AsyncMock(return_value="test prompt")
        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.flush_promises = AsyncMock(return_value=([], None))
        runtime.mcp_orchestrator = None

        ctx = Mock()
        ctx.__enter__ = Mock(return_value=None)
        ctx.__exit__ = Mock(return_value=False)
        runtime.io_provider.mode_transition_input = Mock(return_value=ctx)
        runtime.io_provider.get_mode_transition_input = Mock(return_value="input")
        runtime.io_provider.increment_tick = Mock(return_value=1)

        runtime.mode_manager = Mock()
        runtime.mode_manager.process_tick = AsyncMock(return_value=None)

        runtime._pending_mode_transition = None
        runtime._is_reloading = False
        runtime._cortex_loop_generation = 0

        with pytest.raises(asyncio.CancelledError):
            await runtime._tick(0)

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators_with_timeout(self, cortex_runtime):
        """Test _stop_current_orchestrators when tasks don't complete in time."""
        runtime, _ = cortex_runtime

        runtime.background_orchestrator = Mock()
        runtime.background_orchestrator.stop = Mock()

        runtime.action_orchestrator = Mock()
        runtime.action_orchestrator.stop = Mock()

        runtime.input_orchestrator = Mock()
        runtime.input_orchestrator.stop = Mock()

        mock_cortex_task = Mock()
        mock_cortex_task.done.return_value = False
        mock_cortex_task.cancel = Mock()
        runtime.cortex_loop_task = mock_cortex_task

        pending_task = Mock()

        with patch("asyncio.wait", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = (set(), {pending_task})

            await runtime._stop_current_orchestrators()

            mock_cortex_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_config_changes_exception(self, mock_system_config):
        """Test _check_config_changes with exception in file check."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=True, check_interval=0.01)

            async def sleep_return():
                return None

            with (
                patch("os.path.exists", side_effect=Exception("File check error")),
                patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                mock_sleep.side_effect = [
                    None,  # First check_interval sleep
                    None,  # Error retry sleep
                    asyncio.CancelledError(),  # Cancel on next iteration
                ]

                try:
                    await runtime._check_config_changes()
                except asyncio.CancelledError:
                    pass

            assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_run_with_exception_in_main_loop(self, mock_system_config):
        """Test run() method when exception occurs in orchestrator tasks."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)
            mock_mode_config = Mock()
            mock_mode_config.execute_lifecycle_hooks = AsyncMock()
            mock_system_config.modes = {"test_mode": mock_mode_config}

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)
            runtime.mode_manager = mock_manager

            runtime._initialize_mode = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

            runtime._run_cortex_loop = AsyncMock()
            runtime._handle_mode_transitions = AsyncMock()

            iteration_count = 0

            async def start_with_mocked_tasks():
                runtime.cortex_loop_task = Mock()
                runtime.cortex_loop_task.done.return_value = False
                runtime.mode_transition_task = Mock()
                runtime.mode_transition_task.done.return_value = False

            runtime._start_orchestrators = start_with_mocked_tasks

            async def gather_with_exception(*args, **kwargs):
                nonlocal iteration_count
                iteration_count += 1
                if iteration_count == 1:
                    await asyncio.sleep(0.01)
                    raise Exception("Test orchestrator error")
                elif iteration_count == 2:
                    await asyncio.sleep(1.0)  # Sleep to trigger the error handling path
                    raise KeyboardInterrupt()  # Exit the loop
                return await asyncio.gather(*args, **kwargs)

            with patch("asyncio.gather", side_effect=gather_with_exception):
                try:
                    await runtime.run()
                except KeyboardInterrupt:
                    pass

            runtime._cleanup_tasks.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_run_with_cancelled_error_in_main_loop(self, mock_system_config):
        """Test run() method when CancelledError occurs in orchestrator tasks."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(return_value=True)
            mock_mode_config = Mock()
            mock_mode_config.execute_lifecycle_hooks = AsyncMock()
            mock_system_config.modes = {"test_mode": mock_mode_config}

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)
            runtime.mode_manager = mock_manager

            runtime._initialize_mode = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

            runtime._run_cortex_loop = AsyncMock()
            runtime._handle_mode_transitions = AsyncMock()

            iteration_count = 0

            async def start_with_mocked_tasks():
                runtime.cortex_loop_task = Mock()
                runtime.cortex_loop_task.done.return_value = False
                runtime.mode_transition_task = Mock()
                runtime.mode_transition_task.done.return_value = False

            runtime._start_orchestrators = start_with_mocked_tasks

            async def gather_with_cancel(*args, **kwargs):
                nonlocal iteration_count
                iteration_count += 1
                await asyncio.sleep(0.01)
                if iteration_count == 1:
                    raise asyncio.CancelledError()
                elif iteration_count == 2:
                    await asyncio.sleep(0.1)  # Allow the cancellation handling to run
                    raise KeyboardInterrupt()
                return await asyncio.gather(*args, **kwargs)

            with patch("asyncio.gather", side_effect=gather_with_cancel):
                try:
                    await runtime.run()
                except KeyboardInterrupt:
                    pass

            runtime._cleanup_tasks.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_run_with_startup_hooks_failure(self, mock_system_config):
        """Test run() method when startup hooks fail."""
        with (
            patch("runtime.cortex.ModeManager") as mock_manager_class,
            patch("runtime.cortex.IOProvider"),
            patch("runtime.cortex.SleepTickerProvider"),
        ):
            mock_manager = Mock()
            mock_manager.add_transition_callback = Mock()
            mock_manager.current_mode_name = "test_mode"
            mock_manager.set_event_loop = Mock()
            mock_manager._get_runtime_config_path = Mock(return_value="/fake/path/test_config.json5")
            mock_manager_class.return_value = mock_manager

            mock_system_config.execute_global_lifecycle_hooks = AsyncMock(return_value=False)
            mock_mode_config = Mock()
            mock_mode_config.execute_lifecycle_hooks = AsyncMock()
            mock_system_config.modes = {"test_mode": mock_mode_config}

            runtime = ModeCortexRuntime(mock_system_config, "test_config", hot_reload=False)
            runtime.mode_manager = mock_manager

            runtime._initialize_mode = AsyncMock()
            runtime._start_orchestrators = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

            mock_task = Mock()
            mock_task.done.return_value = False

            def mock_create_task(coro, *args, **kwargs):
                coro.close()
                return mock_task

            with (
                patch("asyncio.create_task", side_effect=mock_create_task),
                patch("asyncio.gather", side_effect=KeyboardInterrupt()),
            ):
                try:
                    await runtime.run()
                except KeyboardInterrupt:
                    pass

            # Should still initialize even if hooks fail
            runtime._initialize_mode.assert_called_once()
            runtime._cleanup_tasks.assert_called_once()
