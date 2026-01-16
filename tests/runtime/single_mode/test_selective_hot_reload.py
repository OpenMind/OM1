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
    """Tests for _detect_config_changes method."""

    @pytest.fixture
    def runtime(self):
        """Create a CortexRuntime instance for testing."""
        with patch("runtime.single_mode.cortex.Fuser"):
            with patch("runtime.single_mode.cortex.ActionOrchestrator"):
                with patch("runtime.single_mode.cortex.SimulatorOrchestrator"):
                    with patch("runtime.single_mode.cortex.BackgroundOrchestrator"):
                        with patch("runtime.single_mode.cortex.IOProvider"):
                            with patch("runtime.single_mode.cortex.ConfigProvider"):
                                with patch(
                                    "runtime.single_mode.cortex.SleepTickerProvider"
                                ):
                                    config = MockRuntimeConfig()
                                    rt = CortexRuntime(
                                        config=config,
                                        config_name="test",
                                        hot_reload=False,
                                    )
                                    return rt

    def test_no_changes_returns_empty_set(self, runtime):
        """Test that identical configs return empty set."""
        llm = MagicMock()
        old_config = MockRuntimeConfig(cortex_llm=llm)
        new_config = MockRuntimeConfig(cortex_llm=llm)

        changed = runtime._detect_config_changes(old_config, new_config)

        assert changed == set()

    def test_detects_system_prompt_base_change(self, runtime):
        """Test detection of system_prompt_base change."""
        old_config = MockRuntimeConfig(system_prompt_base="Old prompt")
        new_config = MockRuntimeConfig(system_prompt_base="New prompt")

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "system_prompt_base" in changed

    def test_detects_system_governance_change(self, runtime):
        """Test detection of system_governance change."""
        old_config = MockRuntimeConfig(system_governance="Old rules")
        new_config = MockRuntimeConfig(system_governance="New rules")

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "system_governance" in changed

    def test_detects_hertz_change(self, runtime):
        """Test detection of hertz change."""
        old_config = MockRuntimeConfig(hertz=1.0)
        new_config = MockRuntimeConfig(hertz=2.0)

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "hertz" in changed

    def test_detects_multiple_safe_field_changes(self, runtime):
        """Test detection of multiple safe field changes."""
        old_config = MockRuntimeConfig(
            system_prompt_base="Old",
            hertz=1.0,
        )
        new_config = MockRuntimeConfig(
            system_prompt_base="New",
            hertz=2.0,
        )

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "system_prompt_base" in changed
        assert "hertz" in changed

    def test_detects_agent_inputs_length_change(self, runtime):
        """Test detection of agent_inputs length change."""
        old_config = MockRuntimeConfig(agent_inputs=[])
        new_config = MockRuntimeConfig(agent_inputs=[MagicMock()])

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "agent_inputs" in changed


    def test_detects_agent_inputs_content_change_same_length(self, runtime):
        """Detect changes when agent_inputs content changes but length stays the same."""
        old_config = MockRuntimeConfig(agent_inputs=[MagicMock(name="old_input")])
        new_config = MockRuntimeConfig(agent_inputs=[MagicMock(name="new_input")])

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "agent_inputs" in changed

    def test_detects_cortex_llm_change(self, runtime):
        """Test detection of cortex_llm instance change."""
        llm1 = MagicMock()
        llm2 = MagicMock()
        old_config = MockRuntimeConfig(cortex_llm=llm1)
        new_config = MockRuntimeConfig(cortex_llm=llm2)

        changed = runtime._detect_config_changes(old_config, new_config)

        assert "cortex_llm" in changed


class TestApplySafeConfigUpdates:
    """Tests for _apply_safe_config_updates method."""

    @pytest.fixture
    def runtime(self):
        """Create a CortexRuntime instance for testing."""
        with patch("runtime.single_mode.cortex.Fuser"):
            with patch("runtime.single_mode.cortex.ActionOrchestrator"):
                with patch("runtime.single_mode.cortex.SimulatorOrchestrator"):
                    with patch("runtime.single_mode.cortex.BackgroundOrchestrator"):
                        with patch("runtime.single_mode.cortex.IOProvider"):
                            with patch("runtime.single_mode.cortex.ConfigProvider"):
                                with patch(
                                    "runtime.single_mode.cortex.SleepTickerProvider"
                                ):
                                    config = MockRuntimeConfig(
                                        system_prompt_base="Old prompt",
                                        hertz=1.0,
                                    )
                                    rt = CortexRuntime(
                                        config=config,
                                        config_name="test",
                                        hot_reload=False,
                                    )
                                    return rt

    def test_updates_system_prompt_base(self, runtime):
        """Test that system_prompt_base is updated correctly."""
        new_config = MockRuntimeConfig(system_prompt_base="New prompt")

        runtime._apply_safe_config_updates(new_config, {"system_prompt_base"})

        assert runtime.config.system_prompt_base == "New prompt"

    def test_updates_hertz(self, runtime):
        """Test that hertz is updated correctly."""
        new_config = MockRuntimeConfig(hertz=5.0)

        runtime._apply_safe_config_updates(new_config, {"hertz"})

        assert runtime.config.hertz == 5.0

    def test_updates_multiple_safe_fields(self, runtime):
        """Test updating multiple safe fields at once."""
        new_config = MockRuntimeConfig(
            system_prompt_base="New prompt",
            system_governance="New rules",
            hertz=3.0,
        )

        runtime._apply_safe_config_updates(
            new_config, {"system_prompt_base", "system_governance", "hertz"}
        )

        assert runtime.config.system_prompt_base == "New prompt"
        assert runtime.config.system_governance == "New rules"
        assert runtime.config.hertz == 3.0

    def test_ignores_unsafe_fields(self, runtime):
        """Test that unsafe fields are not updated."""
        original_inputs = runtime.config.agent_inputs
        new_config = MockRuntimeConfig(agent_inputs=[MagicMock()])

        # Try to update agent_inputs (should be ignored)
        runtime._apply_safe_config_updates(new_config, {"agent_inputs"})

        # agent_inputs should not be changed because it's not in HOT_RELOAD_SAFE_FIELDS
        assert runtime.config.agent_inputs is original_inputs


class TestReloadConfig:
    """Tests for _reload_config method behavior."""

    @pytest.fixture
    def runtime(self):
        """Create a CortexRuntime instance for testing."""
        with patch("runtime.single_mode.cortex.Fuser"):
            with patch("runtime.single_mode.cortex.ActionOrchestrator"):
                with patch("runtime.single_mode.cortex.SimulatorOrchestrator"):
                    with patch("runtime.single_mode.cortex.BackgroundOrchestrator"):
                        with patch("runtime.single_mode.cortex.IOProvider"):
                            with patch("runtime.single_mode.cortex.ConfigProvider"):
                                with patch(
                                    "runtime.single_mode.cortex.SleepTickerProvider"
                                ):
                                    config = MockRuntimeConfig(
                                        system_prompt_base="Old prompt"
                                    )
                                    rt = CortexRuntime(
                                        config=config,
                                        config_name="test",
                                        hot_reload=False,
                                    )
                                    # Set config_path manually since hot_reload is False
                                    rt.config_path = "/tmp/test_config.json5"
                                    rt._full_reload = AsyncMock()
                                    return rt

    @pytest.mark.asyncio
    async def test_safe_changes_do_not_trigger_full_reload(self, runtime):
        """Test that safe field changes don't trigger full reload."""
        # Use same LLM to avoid cortex_llm change detection
        new_config = MockRuntimeConfig(
            system_prompt_base="New prompt",
            cortex_llm=runtime.config.cortex_llm,
        )

        with patch("runtime.single_mode.cortex.load_config", return_value=new_config):
            await runtime._reload_config()

        # Full reload should NOT be called
        runtime._full_reload.assert_not_called()
        # Config should be updated
        assert runtime.config.system_prompt_base == "New prompt"

    @pytest.mark.asyncio
    async def test_unsafe_changes_trigger_full_reload(self, runtime):
        """Test that unsafe field changes trigger full reload."""
        new_config = MockRuntimeConfig(
            agent_inputs=[MagicMock()],
            cortex_llm=runtime.config.cortex_llm,
        )

        with patch("runtime.single_mode.cortex.load_config", return_value=new_config):
            await runtime._reload_config()

        # Full reload SHOULD be called
        runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_changes_trigger_full_reload(self, runtime):
        """Test that mixed safe/unsafe changes trigger full reload."""
        new_config = MockRuntimeConfig(
            system_prompt_base="New prompt",  # safe
            agent_inputs=[MagicMock()],  # unsafe
            cortex_llm=runtime.config.cortex_llm,
        )

        with patch("runtime.single_mode.cortex.load_config", return_value=new_config):
            await runtime._reload_config()

        # Full reload SHOULD be called because of unsafe field
        runtime._full_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_changes_does_nothing(self, runtime):
        """Test that no changes result in no action."""
        # Same config with same LLM instance
        new_config = MockRuntimeConfig(
            system_prompt_base="Old prompt",
            cortex_llm=runtime.config.cortex_llm,
        )

        with patch("runtime.single_mode.cortex.load_config", return_value=new_config):
            await runtime._reload_config()

        runtime._full_reload.assert_not_called()
