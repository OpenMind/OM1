"""Integration tests for the hot-reload module."""

import os
import tempfile

import pytest

from runtime.hot_reload.diff import ConfigDiff, diff_configs, ChangeType
from runtime.hot_reload.strategies import (
    ReloadStrategy,
    categorize_changes,
    get_field_strategy,
    requires_restart,
    validate_field,
)
from runtime.hot_reload.watcher import ConfigFileWatcher
from runtime.hot_reload.manager import HotReloadManager


class TestEndToEndHotReload:
    """End-to-end tests for the hot-reload system."""

    def test_full_diff_to_strategy_flow(self):
        """Test complete flow from diff detection to strategy categorization."""
        old_config = {
            "name": "test_agent",
            "hertz": 1,
            "system_prompt_base": "old prompt",
            "cortex_llm": {"type": "openai", "model": "gpt-3.5"},
        }
        new_config = {
            "name": "test_agent_v2",
            "hertz": 2,
            "system_prompt_base": "new prompt",
            "cortex_llm": {"type": "openai", "model": "gpt-4"},
        }

        diff = diff_configs(old_config, new_config)
        assert diff.has_changes is True

        categorized = categorize_changes(diff.changed_fields)

        assert "system_prompt_base" in categorized[ReloadStrategy.HOT_RELOAD]
        assert "name" in categorized[ReloadStrategy.HOT_RELOAD]
        assert "hertz" in categorized[ReloadStrategy.VALIDATE_FIRST]

    def test_validation_before_hot_reload(self):
        """Test that validation occurs before applying hot-reload changes."""
        changes = {
            "hertz": (1, 2),
            "system_prompt_base": ("old", "new"),
        }

        for field, (old_val, new_val) in changes.items():
            strategy = get_field_strategy(field)

            if strategy == ReloadStrategy.VALIDATE_FIRST:
                assert validate_field(field, old_val, new_val) is True

    def test_validation_rejects_invalid_hertz(self):
        """Test that validation rejects invalid hertz values."""
        changes = {"hertz": (1, -5)}

        for field, (old_val, new_val) in changes.items():
            strategy = get_field_strategy(field)

            if strategy == ReloadStrategy.VALIDATE_FIRST:
                assert validate_field(field, old_val, new_val) is False

    def test_requires_restart_detection(self):
        """Test correct detection of restart-required changes."""
        hot_reload_changes = {
            "name": ("old", "new"),
            "hertz": (1, 2),
            "system_prompt_base": ("old prompt", "new prompt"),
        }
        assert requires_restart(hot_reload_changes) is False

        restart_changes = {
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
        }
        assert requires_restart(restart_changes) is True

        mixed_changes = {
            "name": ("old", "new"),
            "cortex_llm": ({"type": "A"}, {"type": "B"}),
        }
        assert requires_restart(mixed_changes) is True


class TestConfigWatcherWithManager:
    """Tests for ConfigFileWatcher integration."""

    def test_watcher_initialization(self):
        """Test watcher initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert watcher.config_path == config_path
        finally:
            os.unlink(config_path)

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            assert os.path.abspath(config_path) == manager.config_path
        finally:
            os.unlink(config_path)


class TestMultipleFieldChanges:
    """Tests for handling multiple simultaneous field changes."""

    def test_categorize_all_hot_reloadable(self):
        """Test when all changes are hot-reloadable."""
        changes = {
            "name": ("old", "new"),
            "system_prompt_base": ("old prompt", "new prompt"),
            "system_governance": ("old gov", "new gov"),
        }

        assert requires_restart(changes) is False

        categorized = categorize_changes(changes)
        hot_reload_count = len(categorized[ReloadStrategy.HOT_RELOAD])
        assert hot_reload_count >= 2

    def test_categorize_all_restart_required(self):
        """Test when all changes require restart."""
        changes = {
            "cortex_llm": ({}, {"type": "new"}),
            "agent_inputs": ([], [{"type": "X"}]),
            "agent_actions": ([], [{"type": "Y"}]),
        }

        assert requires_restart(changes) is True

    def test_priority_of_restart_over_hot_reload(self):
        """Test that restart takes priority when mixed changes exist."""
        changes = {
            "name": ("old", "new"),
            "cortex_llm": ({}, {"type": "new"}),
        }

        assert requires_restart(changes) is True


class TestNestedConfigChanges:
    """Tests for handling nested configuration changes."""

    def test_nested_llm_config_change(self):
        """Test detecting nested LLM config changes."""
        old_config = {
            "cortex_llm": {
                "type": "openai",
                "config": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                },
            }
        }
        new_config = {
            "cortex_llm": {
                "type": "openai",
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
            }
        }

        diff = diff_configs(old_config, new_config)
        assert diff.has_changes is True

        nested_changes = [
            c for c in diff.changes if "cortex_llm" in c.field_path
        ]
        assert len(nested_changes) > 0

        strategy = get_field_strategy("cortex_llm.config.model")
        assert strategy == ReloadStrategy.RESTART_REQUIRED

    def test_nested_agent_input_change(self):
        """Test detecting nested agent input changes."""
        old_config = {
            "agent_inputs": [
                {"type": "microphone", "enabled": True},
                {"type": "camera", "enabled": False},
            ]
        }
        new_config = {
            "agent_inputs": [
                {"type": "microphone", "enabled": True},
                {"type": "camera", "enabled": True},
            ]
        }

        diff = diff_configs(old_config, new_config)
        assert diff.has_changes is True


class TestErrorHandling:
    """Tests for error handling in hot-reload system."""

    def test_diff_with_none_values(self):
        """Test diffing configs with None values."""
        old_config = {"field": None}
        new_config = {"field": "value"}

        diff = diff_configs(old_config, new_config)
        assert diff.has_changes is True

    def test_diff_type_mismatch(self):
        """Test diffing configs with type mismatches."""
        old_config = {"field": "string"}
        new_config = {"field": 123}

        diff = diff_configs(old_config, new_config)
        assert diff.has_changes is True


class TestThreadSafety:
    """Tests for thread safety in hot-reload system."""

    def test_watcher_has_lock(self):
        """Test that watcher has thread lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert hasattr(watcher, "_lock")
        finally:
            os.unlink(config_path)

    def test_manager_has_lock(self):
        """Test that manager has thread lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            assert hasattr(manager, "_lock")
        finally:
            os.unlink(config_path)
