import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig
from runtime.single_mode.cortex import CortexRuntime


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
        patch(
            "runtime.single_mode.cortex.Fuser", return_value=mock_dependencies["fuser"]
        ),
        patch(
            "runtime.single_mode.cortex.ActionOrchestrator",
            return_value=mock_dependencies["action_orchestrator"],
        ),
        patch(
            "runtime.single_mode.cortex.SimulatorOrchestrator",
            return_value=mock_dependencies["simulator_orchestrator"],
        ),
        patch(
            "runtime.single_mode.cortex.SleepTickerProvider",
            return_value=mock_dependencies["sleep_ticker_provider"],
        ),
        patch(
            "runtime.single_mode.cortex.BackgroundOrchestrator",
            return_value=mock_dependencies["background_orchestrator"],
        ),
    ):
        return CortexRuntime(mock_config, "test_config"), mock_dependencies


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
        "runtime.single_mode.cortex.InputOrchestrator",
        return_value=mocks["input_orchestrator"],
    ):
        mocks["input_orchestrator"].listen = AsyncMock()
        task = await cortex_runtime._start_input_listeners()

        assert isinstance(task, asyncio.Task)
        mocks["input_orchestrator"].listen.assert_called_once()


@pytest.mark.asyncio
async def test_run_full_runtime(runtime):
    cortex_runtime, _ = runtime

    cortex_runtime._start_orchestrators = AsyncMock()
    cortex_runtime._cleanup_tasks = AsyncMock()

    async def mock_cortex_loop():
        await asyncio.sleep(0.01)
        return

    cortex_runtime._run_cortex_loop = AsyncMock(side_effect=mock_cortex_loop)

    try:
        await asyncio.wait_for(cortex_runtime.run(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    cortex_runtime._start_orchestrators.assert_called_once()
    cortex_runtime._run_cortex_loop.assert_called_once()


class TestCortexRuntimeHotReload:
    """Test cases for hot reload functionality in CortexRuntime."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing hot reload."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            f.write('{"test": "config"}')
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_hot_reload_initialization_enabled(self, mock_config, mock_dependencies):
        """Test hot reload initialization when enabled."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=30.0
            )

            assert runtime.hot_reload is True
            assert runtime.check_interval == 30.0
            assert runtime.last_modified == 1234567890.0
            assert runtime.config_path.endswith(".runtime.json5")

    def test_hot_reload_initialization_disabled(self, mock_config, mock_dependencies):
        """Test hot reload initialization when disabled."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=False)

            assert runtime.hot_reload is False
            assert runtime.last_modified == 0.0

    def test_get_file_mtime_existing_file(
        self, mock_config, mock_dependencies, temp_config_file
    ):
        """Test getting modification time of existing file."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)
            runtime.config_path = temp_config_file

            mtime = runtime._get_file_mtime()
            assert mtime > 0

    def test_get_file_mtime_nonexistent_file(self, mock_config, mock_dependencies):
        """Test getting modification time of non-existent file."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)
            runtime.config_path = "/nonexistent/file.json5"

            mtime = runtime._get_file_mtime()
            assert mtime == 0.0

    @pytest.mark.asyncio
    async def test_check_config_changes_file_changed(
        self, mock_config, mock_dependencies, temp_config_file
    ):
        """Test config change detection when file is modified."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.config_path = temp_config_file
            runtime.last_modified = 1.0
            runtime._last_raw_config = {"hertz": 1}
            runtime._read_raw_config = Mock(return_value={"hertz": 1, "agent_inputs": [{"type": "new"}]})

            runtime._full_reload = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._full_reload.assert_called_once()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_check_config_changes_no_change(self, mock_config, mock_dependencies):
        """Test config change detection when file is not modified."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
            patch("os.path.getmtime", return_value=1234567890.0),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.last_modified = 1234567890.0

            runtime._full_reload = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._full_reload.assert_not_called()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_full_reload_success(self, mock_config, mock_dependencies):
        """Test successful full config reload."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
            patch("runtime.single_mode.cortex.load_config") as mock_load_config,
        ):
            new_mock_config = Mock(spec=RuntimeConfig)
            new_mock_config.hertz = 20.0
            mock_load_config.return_value = new_mock_config

            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._start_orchestrators = AsyncMock()

            await runtime._full_reload()

            mock_load_config.assert_called_once()
            runtime._stop_current_orchestrators.assert_called_once()
            runtime._start_orchestrators.assert_called_once()

            assert runtime.config == new_mock_config

    @pytest.mark.asyncio
    async def test_full_reload_no_config_name(self, mock_config, mock_dependencies):
        """Test full reload with no config name."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
            patch("runtime.single_mode.cortex.load_config") as mock_load_config,
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)
            runtime.config_name = ""

            runtime._stop_current_orchestrators = AsyncMock()

            await runtime._full_reload()

            mock_load_config.assert_not_called()
            runtime._stop_current_orchestrators.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_reload_failure(self, mock_config, mock_dependencies):
        """Test full reload failure handling."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.load_config",
                side_effect=Exception("Load failed"),
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)

            runtime._stop_current_orchestrators = AsyncMock()

            await runtime._full_reload()

            runtime._stop_current_orchestrators.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_current_orchestrators(self, mock_config, mock_dependencies):
        """Test stopping current orchestrators for hot reload."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)

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
    async def test_run_with_hot_reload_enabled(self, mock_config, mock_dependencies):
        """Test run method with hot reload enabled."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=1.0
            )

            runtime._start_orchestrators = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

            async def mock_cortex_loop():
                await asyncio.sleep(0.01)
                return

            async def mock_config_watcher():
                await asyncio.sleep(0.01)
                return

            runtime._run_cortex_loop = AsyncMock(side_effect=mock_cortex_loop)
            runtime._check_config_changes = AsyncMock(side_effect=mock_config_watcher)

            try:
                await asyncio.wait_for(runtime.run(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            assert runtime.config_watcher_task is not None
            runtime._check_config_changes.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_none_cortex_loop_task_on_reload(
        self, mock_config, mock_dependencies
    ):
        """Test run() handles cortex_loop_task being None during config reload."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=False)
            runtime._start_orchestrators = AsyncMock()
            runtime._cleanup_tasks = AsyncMock()

            async def mock_cortex_loop():
                # Simulate reload: task is set to None then cancelled
                runtime.cortex_loop_task = None
                raise asyncio.CancelledError()

            runtime._run_cortex_loop = AsyncMock(side_effect=mock_cortex_loop)

            try:
                await asyncio.wait_for(runtime.run(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    @pytest.mark.asyncio
    async def test_cleanup_tasks_with_config_watcher(
        self, mock_config, mock_dependencies
    ):
        """Test cleanup includes config watcher task when hot reload is enabled."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=True)

            # Create mock config watcher task
            mock_config_watcher = Mock()
            mock_config_watcher.done.return_value = False
            mock_config_watcher.cancel = Mock()
            runtime.config_watcher_task = mock_config_watcher

            with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
                await runtime._cleanup_tasks()

                mock_config_watcher.cancel.assert_called_once()
                mock_gather.assert_called_once()


class TestSelectiveHotReload:
    """Test cases for selective field-based hot reload in CortexRuntime."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            f.write('{"test": "config"}')
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_detect_changed_fields_safe_only(self):
        """Only safe fields changed → all in 'safe' set."""
        old = {"hertz": 1, "system_prompt_base": "old", "agent_inputs": []}
        new = {"hertz": 2, "system_prompt_base": "new", "agent_inputs": []}
        result = CortexRuntime._detect_changed_fields(old, new)
        assert result["safe"] == {"hertz", "system_prompt_base"}
        assert result["unsafe"] == set()

    def test_detect_changed_fields_unsafe_only(self):
        """Only unsafe fields changed → all in 'unsafe' set."""
        old = {"hertz": 1, "agent_inputs": [{"type": "A"}]}
        new = {"hertz": 1, "agent_inputs": [{"type": "B"}]}
        result = CortexRuntime._detect_changed_fields(old, new)
        assert result["safe"] == set()
        assert result["unsafe"] == {"agent_inputs"}

    def test_detect_changed_fields_mixed(self):
        """Both safe and unsafe changed → both sets populated."""
        old = {"hertz": 1, "agent_inputs": [], "system_governance": "old"}
        new = {"hertz": 2, "agent_inputs": [{"type": "X"}], "system_governance": "new"}
        result = CortexRuntime._detect_changed_fields(old, new)
        assert result["safe"] == {"hertz", "system_governance"}
        assert result["unsafe"] == {"agent_inputs"}

    def test_detect_changed_fields_no_changes(self):
        """Identical configs → both sets empty."""
        cfg = {"hertz": 1, "system_prompt_base": "same", "agent_inputs": []}
        result = CortexRuntime._detect_changed_fields(cfg, cfg.copy())
        assert result["safe"] == set()
        assert result["unsafe"] == set()

    def test_detect_changed_fields_new_key_added(self):
        """A new key appears in the new config."""
        old = {"hertz": 1}
        new = {"hertz": 1, "cortex_llm": {"type": "OpenAI"}}
        result = CortexRuntime._detect_changed_fields(old, new)
        assert "cortex_llm" in result["unsafe"]

    @pytest.mark.asyncio
    async def test_apply_safe_reload_prompts(self, mock_config, mock_dependencies):
        """Safe reload updates prompt fields in-place on RuntimeConfig."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=False)
            runtime.config.system_prompt_base = "old prompt"
            runtime.config.system_governance = "old gov"

            new_raw = {
                "system_prompt_base": "new prompt",
                "system_governance": "new gov",
            }
            await runtime._apply_safe_reload(
                new_raw, {"system_prompt_base", "system_governance"}
            )

            assert runtime.config.system_prompt_base == "new prompt"
            assert runtime.config.system_governance == "new gov"

    @pytest.mark.asyncio
    async def test_apply_safe_reload_hertz(self, mock_config, mock_dependencies):
        """Safe reload updates hertz in-place."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=False)
            runtime.config.hertz = 1.0

            await runtime._apply_safe_reload({"hertz": 5.0}, {"hertz"})
            assert runtime.config.hertz == 5.0

    @pytest.mark.asyncio
    async def test_apply_safe_reload_invalid_hertz(self, mock_config, mock_dependencies):
        """Invalid hertz value is rejected during safe reload."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(mock_config, "test_config", hot_reload=False)
            runtime.config.hertz = 1.0

            # Zero hertz → rejected
            await runtime._apply_safe_reload({"hertz": 0}, {"hertz"})
            assert runtime.config.hertz == 1.0

            # Negative hertz → rejected
            await runtime._apply_safe_reload({"hertz": -5}, {"hertz"})
            assert runtime.config.hertz == 1.0

    @pytest.mark.asyncio
    async def test_check_config_safe_change_skips_full_reload(
        self, mock_config, mock_dependencies, temp_config_file
    ):
        """Safe-only field change does NOT trigger _full_reload."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.config_path = temp_config_file
            runtime.last_modified = 1.0
            runtime.config.system_prompt_base = "old"

            runtime._last_raw_config = {"system_prompt_base": "old", "hertz": 1}
            runtime._read_raw_config = Mock(
                return_value={"system_prompt_base": "new", "hertz": 1}
            )
            runtime._full_reload = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())
            try:
                await asyncio.sleep(0.2)
                task.cancel()
                runtime._full_reload.assert_not_called()
                assert runtime.config.system_prompt_base == "new"
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_check_config_no_field_changes_skips_reload(
        self, mock_config, mock_dependencies, temp_config_file
    ):
        """File touched but content identical → no reload at all."""
        with (
            patch(
                "runtime.single_mode.cortex.Fuser",
                return_value=mock_dependencies["fuser"],
            ),
            patch(
                "runtime.single_mode.cortex.ActionOrchestrator",
                return_value=mock_dependencies["action_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SimulatorOrchestrator",
                return_value=mock_dependencies["simulator_orchestrator"],
            ),
            patch(
                "runtime.single_mode.cortex.SleepTickerProvider",
                return_value=mock_dependencies["sleep_ticker_provider"],
            ),
            patch(
                "runtime.single_mode.cortex.BackgroundOrchestrator",
                return_value=mock_dependencies["background_orchestrator"],
            ),
        ):
            runtime = CortexRuntime(
                mock_config, "test_config", hot_reload=True, check_interval=0.1
            )
            runtime.config_path = temp_config_file
            runtime.last_modified = 1.0

            same_config = {"system_prompt_base": "same", "hertz": 1}
            runtime._last_raw_config = same_config.copy()
            runtime._read_raw_config = Mock(return_value=same_config.copy())
            runtime._full_reload = AsyncMock()
            runtime._apply_safe_reload = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())
            try:
                await asyncio.sleep(0.2)
                task.cancel()
                runtime._full_reload.assert_not_called()
                runtime._apply_safe_reload.assert_not_called()
            except asyncio.CancelledError:
                pass
