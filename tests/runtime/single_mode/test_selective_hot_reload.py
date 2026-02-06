"""
Tests for selective hot-reload functionality in CortexRuntime.

These tests verify that configuration changes are handled efficiently:
- Safe fields (prompts, hertz) are updated without restarting orchestrators
- Unsafe fields (inputs, actions, LLM) trigger a full restart
"""

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.single_mode.cortex import (
    HOT_RELOAD_SAFE_FIELDS,
    CortexRuntime,
)


@dataclass
class MockRuntimeConfig:
    """Mock RuntimeConfig for testing."""

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
def mock_cortex_deps():
    """Mock all CortexRuntime dependencies."""
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
    def runtime(self, mock_cortex_deps):
        """Create a CortexRuntime instance for testing."""
        return _make_runtime(mock_cortex_deps)

    def test_no_changes_returns_empty_set(self, runtime):
        """Test that identical raw configs return empty set."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert changed == set()

    def test_detects_system_prompt_base_change(self, runtime):
        """Test detection of system_prompt_base change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "system_prompt_base" in changed

    def test_detects_system_governance_change(self, runtime):
        """Test detection of system_governance change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_governance": "New rules"}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "system_governance" in changed

    def test_detects_hertz_change(self, runtime):
        """Test detection of hertz change."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "hertz": 2.0}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "hertz" in changed

    def test_detects_multiple_safe_field_changes(self, runtime):
        """Test detection of multiple safe field changes."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New", "hertz": 2.0}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "system_prompt_base" in changed
        assert "hertz" in changed

    def test_detects_agent_inputs_change(self, runtime):
        """Test detection of agent_inputs change."""
        old_raw = {**BASE_RAW_CONFIG, "agent_inputs": []}
        new_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "NewInput"}]}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "agent_inputs" in changed

    def test_detects_agent_inputs_content_change_same_length(self, runtime):
        """Test detection when agent_inputs content changes but length stays same."""
        old_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "InputA"}]}
        new_raw = {**BASE_RAW_CONFIG, "agent_inputs": [{"type": "InputB"}]}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "agent_inputs" in changed

    def test_detects_cortex_llm_change(self, runtime):
        """Test detection of cortex_llm config change."""
        old_raw = {**BASE_RAW_CONFIG, "cortex_llm": {"type": "OpenAILLM"}}
        new_raw = {**BASE_RAW_CONFIG, "cortex_llm": {"type": "GeminiLLM"}}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "cortex_llm" in changed

    def test_detects_new_field_added(self, runtime):
        """Test detection when a new field is added to config."""
        old_raw = {**BASE_RAW_CONFIG}
        new_raw = {**BASE_RAW_CONFIG, "new_field": "value"}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "new_field" in changed

    def test_detects_field_removed(self, runtime):
        """Test detection when a field is removed from config."""
        old_raw = {**BASE_RAW_CONFIG, "extra_field": "value"}
        new_raw = {**BASE_RAW_CONFIG}

        changed = runtime._detect_config_changes(old_raw, new_raw)

        assert "extra_field" in changed


class TestReloadConfig:
    """Tests for _reload_config method behavior."""

    @pytest.fixture
    def runtime(self, mock_cortex_deps):
        """Create a CortexRuntime instance for testing."""
        rt = _make_runtime(mock_cortex_deps, system_prompt_base="Old prompt")
        rt.config_path = "/tmp/test_config.json5"
        rt._raw_config = {**BASE_RAW_CONFIG, "system_prompt_base": "Old prompt"}
        rt._full_reload = AsyncMock()  # type: ignore[method-assign]
        return rt

    @pytest.mark.asyncio
    async def test_safe_changes_do_not_trigger_full_reload(self, runtime):
        """Test that safe field changes don't trigger full reload."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        with patch.object(runtime, "_read_raw_config", return_value=new_raw):
            await runtime._reload_config()

        runtime._full_reload.assert_not_called()
        assert runtime.config.system_prompt_base == "New prompt"

    @pytest.mark.asyncio
    async def test_unsafe_changes_trigger_full_reload(self, runtime):
        """Test that unsafe field changes trigger full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "Old prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await runtime._reload_config()

        runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_changes_trigger_full_reload(self, runtime):
        """Test that mixed safe/unsafe changes trigger full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "New prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await runtime._reload_config()

        runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_changes_does_nothing(self, runtime):
        """Test that no changes result in no action."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "Old prompt"}

        with patch.object(runtime, "_read_raw_config", return_value=new_raw):
            await runtime._reload_config()

        runtime._full_reload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_config_updated_after_safe_reload(self, runtime):
        """Test that _raw_config is updated after a safe field change."""
        new_raw = {**BASE_RAW_CONFIG, "system_prompt_base": "New prompt"}

        with patch.object(runtime, "_read_raw_config", return_value=new_raw):
            await runtime._reload_config()

        assert runtime._raw_config["system_prompt_base"] == "New prompt"

    @pytest.mark.asyncio
    async def test_raw_config_updated_after_full_reload(self, runtime):
        """Test that _raw_config is updated after a full reload."""
        new_raw = {
            **BASE_RAW_CONFIG,
            "system_prompt_base": "Old prompt",
            "agent_inputs": [{"type": "NewInput"}],
        }

        with (
            patch.object(runtime, "_read_raw_config", return_value=new_raw),
            patch("runtime.single_mode.cortex.load_config") as mock_load,
        ):
            mock_load.return_value = MockRuntimeConfig()
            await runtime._reload_config()

        assert runtime._raw_config["agent_inputs"] == [{"type": "NewInput"}]
