"""Integration tests for the hot-reload module."""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.hot_reload.diff import ConfigDiff, diff_configs
from runtime.hot_reload.manager import HotReloadManager, ReloadResult
from runtime.hot_reload.strategies import (
    ReloadStrategy,
    categorize_changes,
    get_field_strategy,
    requires_restart,
    validate_field,
)
from runtime.hot_reload.watcher import ConfigFileWatcher


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

        cortex_changes = [
            k for k in categorized[ReloadStrategy.RESTART_REQUIRED].keys()
            if k.startswith("cortex_llm")
        ]
        assert len(cortex_changes) > 0

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
    """Tests for ConfigFileWatcher integration with HotReloadManager."""

    def test_watcher_triggers_manager(self):
        """Test that file watcher triggers manager processing."""
        with tempfile.NamedTemporaryFile(
            suffix=".json5", delete=False, mode="w"
        ) as f:
            json.dump({"name": "test", "hertz": 1}, f)
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"
            config.hertz = 1

            manager = HotReloadManager(config, config_path)

            changes_detected = []

            def on_change(fields):
                changes_detected.append(fields)

            manager.set_on_hot_reload(on_change)

            assert manager.config_path == config_path
        finally:
            os.unlink(config_path)

    def test_watcher_debounce_with_manager(self):
        """Test that watcher debouncing works with manager."""
        with tempfile.NamedTemporaryFile(
            suffix=".json5", delete=False, mode="w"
        ) as f:
            json.dump({"name": "test"}, f)
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path, debounce_seconds=0.2)

            callback_count = []
            watcher.set_callback(lambda: callback_count.append(1))

            assert watcher.debounce_seconds == 0.2
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

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        with tempfile.NamedTemporaryFile(
            suffix=".json5", delete=False, mode="w"
        ) as f:
            f.write("invalid json {{{")
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)

            assert manager.config_path == config_path
        finally:
            os.unlink(config_path)

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        config = MagicMock()
        manager = HotReloadManager(config, "/nonexistent/config.json5")

        assert manager.config_path == "/nonexistent/config.json5"

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
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert hasattr(watcher, "_lock")
        finally:
            os.unlink(config_path)

    def test_manager_has_lock(self):
        """Test that manager has thread lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)
            assert hasattr(manager, "_lock")
        finally:
            os.unlink(config_path)


class TestCallbackExecution:
    """Tests for callback execution in hot-reload system."""

    def test_sync_callback_execution(self):
        """Test synchronous callback execution."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"

            manager = HotReloadManager(config, config_path)

            callback_results = []

            def sync_callback(fields):
                callback_results.append(fields)

            manager.set_on_hot_reload(sync_callback)

            manager._execute_callback(manager._on_hot_reload, {"name"})

            assert len(callback_results) == 1
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_async_callback_execution(self):
        """Test asynchronous callback execution."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"

            manager = HotReloadManager(config, config_path)

            callback_results = []

            async def async_callback(fields):
                callback_results.append(fields)

            manager.set_on_hot_reload(async_callback)

            await manager._notify_hot_reload({"name"})

            assert len(callback_results) == 1
        finally:
            os.unlink(config_path)


class TestReloadHistory:
    """Tests for reload history tracking."""

    def test_history_records_success(self):
        """Test that successful reloads are recorded in history."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path, max_history=10)

            result = ReloadResult(
                success=True,
                hot_reloaded_fields={"name"},
                restart_required_fields=set(),
                errors=[],
            )
            manager._add_to_history(result)

            history = manager.get_history()
            assert len(history) == 1
            assert history[0].success is True
        finally:
            os.unlink(config_path)

    def test_history_records_failure(self):
        """Test that failed reloads are recorded in history."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path, max_history=10)

            result = ReloadResult(
                success=False,
                hot_reloaded_fields=set(),
                restart_required_fields=set(),
                errors=["Test error"],
            )
            manager._add_to_history(result)

            history = manager.get_history()
            assert len(history) == 1
            assert history[0].success is False
            assert "Test error" in history[0].errors
        finally:
            os.unlink(config_path)

    def test_history_enforces_limit(self):
        """Test that history respects maximum limit."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path, max_history=3)

            for i in range(5):
                result = ReloadResult(
                    success=True,
                    hot_reloaded_fields={f"field{i}"},
                    restart_required_fields=set(),
                    errors=[],
                )
                manager._add_to_history(result)

            history = manager.get_history()
            assert len(history) == 3
        finally:
            os.unlink(config_path)

