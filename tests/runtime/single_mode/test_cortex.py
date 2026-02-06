import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig
from runtime.single_mode.cortex import (
    HOT_RELOAD_SAFE_FIELDS,
    CortexRuntime,
)


@dataclass
class MockRuntimeConfig:
    """Mock RuntimeConfig for testing selective hot-reload."""

    version: str = "1.0.0"
    hertz: float = 1.0
    name: str = "test"
    system_prompt_base: str = "You are a helpful robot."
    system_governance: str = "Be safe and ethical."
    system_prompt_examples: str = "Example: Hello!"
    agent_inputs: Optional[List] = None
    cortex_llm: Optional[MagicMock] = None
    simulators: Optional[List] = None
    agent_actions: Optional[List] = None
    backgrounds: Optional[List] = None
    mode: Optional[str] = None
    api_key: Optional[str] = None
    robot_ip: Optional[str] = None
    URID: Optional[str] = None
    unitree_ethernet: Optional[str] = None
    action_execution_mode: Optional[str] = None
    action_dependencies: Optional[dict] = None

    def __post_init__(self):
        if self.agent_inputs is None:
            self.agent_inputs = []
        if self.simulators is None:
            self.simulators = []
        if self.agent_actions is None:
            self.agent_actions = []
        if self.backgrounds is None:
            self.backgrounds = []
        if self.cortex_llm is None:
            self.cortex_llm = MagicMock()


BASE_RAW_CONFIG = {
    "version": "1.0.0",
    "hertz": 1.0,
    "name": "test",
    "system_prompt_base": "You are a helpful robot.",
    "system_governance": "Be safe and ethical.",
    "system_prompt_examples": "Example: Hello!",
    "agent_inputs": [],
    "cortex_llm": {"type": "OpenAILLM"},
    "simulators": [],
    "agent_actions": [],
    "backgrounds": [],
}


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

            runtime._reload_config = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._reload_config.assert_called_once()
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

            runtime._reload_config = AsyncMock()

            task = asyncio.create_task(runtime._check_config_changes())

            try:
                await asyncio.sleep(0.2)
                task.cancel()

                runtime._reload_config.assert_not_called()
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_reload_config_success(self, mock_config, mock_dependencies):
        """Test successful config reload with unsafe field change."""
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

            # Set up raw configs with an unsafe field change to trigger full reload
            runtime._raw_config = {"agent_inputs": []}
            runtime._read_raw_config = Mock(
                return_value={"agent_inputs": [{"type": "NewInput"}]}
            )

            runtime._stop_current_orchestrators = AsyncMock()
            runtime._start_orchestrators = AsyncMock()

            await runtime._reload_config()

            mock_load_config.assert_called_once()
            runtime._stop_current_orchestrators.assert_called_once()
            runtime._start_orchestrators.assert_called_once()

            assert runtime.config == new_mock_config

    @pytest.mark.asyncio
    async def test_reload_config_no_config_name(self, mock_config, mock_dependencies):
        """Test config reload with no config name."""
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

            await runtime._reload_config()

            mock_load_config.assert_not_called()
            runtime._stop_current_orchestrators.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_config_failure(self, mock_config, mock_dependencies):
        """Test config reload failure handling."""
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

            await runtime._reload_config()

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


@pytest.fixture
def mock_cortex_deps():
    """Mock all CortexRuntime dependencies for selective hot-reload tests."""
    with (
        patch("runtime.single_mode.cortex.Fuser"),
        patch("runtime.single_mode.cortex.ActionOrchestrator"),
        patch("runtime.single_mode.cortex.SimulatorOrchestrator"),
        patch("runtime.single_mode.cortex.BackgroundOrchestrator"),
        patch("runtime.single_mode.cortex.IOProvider"),
        patch("runtime.single_mode.cortex.ConfigProvider"),
        patch("runtime.single_mode.cortex.SleepTickerProvider"),
    ):
        yield


def _make_runtime(mock_cortex_deps, **config_overrides):
    """Helper to create a CortexRuntime with mock config."""
    config = MockRuntimeConfig(**config_overrides)
    rt = CortexRuntime(
        config=config,  # type: ignore[arg-type]
        config_name="test",
        hot_reload=False,
    )
    return rt


class TestHotReloadSafeFields:
    """Tests for HOT_RELOAD_SAFE_FIELDS constant."""

    def test_safe_fields_contains_expected_fields(self):
        """Verify that safe fields include system prompts and hertz."""
        assert "system_prompt_base" in HOT_RELOAD_SAFE_FIELDS
        assert "system_governance" in HOT_RELOAD_SAFE_FIELDS
        assert "system_prompt_examples" in HOT_RELOAD_SAFE_FIELDS
        assert "hertz" in HOT_RELOAD_SAFE_FIELDS

    def test_safe_fields_does_not_contain_unsafe_fields(self):
        """Verify that unsafe fields are not in safe fields set."""
        assert "agent_inputs" not in HOT_RELOAD_SAFE_FIELDS
        assert "agent_actions" not in HOT_RELOAD_SAFE_FIELDS
        assert "cortex_llm" not in HOT_RELOAD_SAFE_FIELDS
        assert "simulators" not in HOT_RELOAD_SAFE_FIELDS
        assert "backgrounds" not in HOT_RELOAD_SAFE_FIELDS


class TestDetectConfigChanges:
    """Tests for _detect_config_changes method using raw JSON dicts."""

    @pytest.fixture
    def selective_runtime(self, mock_cortex_deps):
        """Create a CortexRuntime instance for testing."""
        return _make_runtime(mock_cortex_deps)

    def test_no_changes_returns_empty_set(self, selective_runtime):
        """Test that identical raw configs return empty set."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert changed == set()

    def test_detects_system_prompt_base_change(self, selective_runtime):
        """Test detection of system_prompt_base change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "system_prompt_base" in changed

    def test_detects_system_governance_change(self, selective_runtime):
        """Test detection of system_governance change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_governance": "New rules"}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "system_governance" in changed

    def test_detects_hertz_change(self, selective_runtime):
        """Test detection of hertz change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "hertz": 2.0}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "hertz" in changed

    def test_detects_multiple_safe_field_changes(self, selective_runtime):
        """Test detection of multiple safe field changes."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New", "hertz": 2.0}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "system_prompt_base" in changed
        assert "hertz" in changed

    def test_detects_agent_inputs_change(self, selective_runtime):
        """Test detection of agent_inputs change."""
        old_raw = {**BASE_RAW_CONFIG, "agent_inputs": []}
        new_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "NewInput"}]}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "agent_inputs" in changed

    def test_detects_agent_inputs_content_change_same_length(self, selective_runtime):
        """Test detection when agent_inputs content changes but length stays same."""
        old_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "InputA"}]}
        new_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "InputB"}]}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "agent_inputs" in changed

    def test_detects_cortex_llm_change(self, selective_runtime):
        """Test detection of cortex_llm config change."""
        old_raw = {**BASE_RAW_CONFIG, "cortex_llm": {"type": "OpenAILLM"}}
        new_raw = {**BASE_RAW_CONFIG, "cortex_llm": {"type": "GeminiLLM"}}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "cortex_llm" in changed

    def test_detects_new_field_added(self, selective_runtime):
        """Test detection when a new field is added to config."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "new_field": "value"}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "new_field" in changed

    def test_detects_field_removed(self, selective_runtime):
        """Test detection when a field is removed from config."""
        old_raw = {**BASE_RAW_CONFIG, "extra_field": "value"}
        new_raw = {**BASE_RAW_CONFIG}

        changed = selective_runtime._detect_config_changes(old_raw, new_raw)

        assert "extra_field" in changed


class TestSelectiveReloadConfig:
    """Tests for selective _reload_config method behavior."""

    @pytest.fixture
    def selective_runtime(self, mock_cortex_deps):
        """Create a CortexRuntime instance for testing."""
        rt = _make_runtime(mock_cortex_deps, system_prompt_base="Old prompt")
        rt.config_path = "/tmp/test_config.json5"
        rt._raw_config = {**BASE_RAW_CONFIG, "system_prompt_base": "Old prompt"}
        rt._full_reload = AsyncMock()  # type: ignore[method-assign]
        return rt

    @pytest.mark.asyncio
    async def test_safe_changes_do_not_trigger_full_reload(self, selective_runtime):
        """Test that safe field changes don't trigger full reload."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        with patch.object(selective_runtime, "_read_raw_config", return_value=new_raw):
            await selective_runtime._reload_config()

        selective_runtime._full_reload.assert_not_called()
        assert selective_runtime.config.system_prompt_base == "New prompt"

    @pytest.mark.asyncio
    async def test_unsafe_changes_trigger_full_reload(self, selective_runtime):
        """Test that unsafe field changes trigger full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "Old prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(selective_runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await selective_runtime._reload_config()

        selective_runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_changes_trigger_full_reload(self, selective_runtime):
        """Test that mixed safe/unsafe changes trigger full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "New prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(selective_runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await selective_runtime._reload_config()

        selective_runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_changes_does_nothing(self, selective_runtime):
        """Test that no changes result in no action."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "Old prompt"}

        with patch.object(selective_runtime, "_read_raw_config", return_value=new_raw):
            await selective_runtime._reload_config()

        selective_runtime._full_reload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_config_updated_after_safe_reload(self, selective_runtime):
        """Test that _raw_config is updated after a safe field change."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        with patch.object(selective_runtime, "_read_raw_config", return_value=new_raw):
            await selective_runtime._reload_config()

        assert selective_runtime._raw_config["system_prompt_base"] == "New prompt"

    @pytest.mark.asyncio
    async def test_raw_config_updated_after_full_reload(self, selective_runtime):
        """Test that _raw_config is updated after a full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "Old prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(selective_runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await selective_runtime._reload_config()

        assert selective_runtime._raw_config["agent_inputs"] == [{"type": "NewInput"}]
