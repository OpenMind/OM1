"""Tests for the manager module."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.hot_reload.manager import HotReloadManager, ReloadResult
from runtime.hot_reload.strategies import ReloadStrategy


class TestReloadResult:
    """Tests for ReloadResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful reload result."""
        result = ReloadResult(
            success=True,
            hot_reloaded_fields={"hertz", "name"},
            restart_required_fields=set(),
            errors=[],
            timestamp=datetime.now(),
        )
        assert result.success is True
        assert "hertz" in result.hot_reloaded_fields
        assert len(result.errors) == 0

    def test_failed_result(self):
        """Test creating a failed reload result."""
        result = ReloadResult(
            success=False,
            hot_reloaded_fields=set(),
            restart_required_fields={"cortex_llm"},
            errors=["Failed to reload cortex_llm"],
            timestamp=datetime.now(),
        )
        assert result.success is False
        assert "cortex_llm" in result.restart_required_fields
        assert len(result.errors) == 1

    def test_result_with_mixed_fields(self):
        """Test result with both hot-reloaded and restart-required fields."""
        result = ReloadResult(
            success=True,
            hot_reloaded_fields={"hertz", "name"},
            restart_required_fields={"cortex_llm"},
            errors=[],
            timestamp=datetime.now(),
        )
        assert len(result.hot_reloaded_fields) == 2
        assert len(result.restart_required_fields) == 1


class TestHotReloadManager:
    """Tests for HotReloadManager class."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)
            assert manager.config == config
            assert manager.config_path == config_path
            assert manager._is_running is False
        finally:
            os.unlink(config_path)

    def test_manager_start_stop(self):
        """Test manager can be started and stopped."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)

            manager.start()
            assert manager._is_running is True

            manager.stop()
            assert manager._is_running is False
        finally:
            os.unlink(config_path)

    def test_manager_get_history(self):
        """Test manager tracks reload history."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)

            history = manager.get_history()
            assert isinstance(history, list)
            assert len(history) == 0
        finally:
            os.unlink(config_path)

    def test_manager_history_limit(self):
        """Test manager limits history size."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path, max_history=5)

            for i in range(10):
                manager._add_to_history(
                    ReloadResult(
                        success=True,
                        hot_reloaded_fields={f"field{i}"},
                        restart_required_fields=set(),
                        errors=[],
                        timestamp=datetime.now(),
                    )
                )

            history = manager.get_history()
            assert len(history) <= 5
        finally:
            os.unlink(config_path)

    def test_manager_context_manager(self):
        """Test manager works as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            with HotReloadManager(config, config_path) as manager:
                manager.start()
                assert manager._is_running is True

            assert manager._is_running is False
        finally:
            os.unlink(config_path)

    def test_manager_set_on_hot_reload(self):
        """Test setting hot reload callback."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)

            callback = MagicMock()
            manager.set_on_hot_reload(callback)
            assert manager._on_hot_reload == callback
        finally:
            os.unlink(config_path)

    def test_manager_set_on_restart_required(self):
        """Test setting restart required callback."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            config = MagicMock()
            manager = HotReloadManager(config, config_path)

            callback = MagicMock()
            manager.set_on_restart_required(callback)
            assert manager._on_restart_required == callback
        finally:
            os.unlink(config_path)


class TestHotReloadManagerApplyChanges:
    """Tests for HotReloadManager change application."""

    def test_apply_hot_reloadable_field(self):
        """Test applying a hot-reloadable field change."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "old_name", "hertz": 1}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "old_name"
            config.hertz = 1

            manager = HotReloadManager(config, config_path)

            changes = {"name": ("old_name", "new_name")}
            result = manager._apply_changes(changes)

            assert result.success is True
            assert "name" in result.hot_reloaded_fields
        finally:
            os.unlink(config_path)

    def test_apply_restart_required_field(self):
        """Test handling restart-required field change."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"cortex_llm": {"type": "A"}}')
            config_path = f.name

        try:
            config = MagicMock()
            config.cortex_llm = {"type": "A"}

            manager = HotReloadManager(config, config_path)

            changes = {"cortex_llm": ({"type": "A"}, {"type": "B"})}
            result = manager._apply_changes(changes)

            assert "cortex_llm" in result.restart_required_fields
        finally:
            os.unlink(config_path)

    def test_apply_mixed_changes(self):
        """Test applying mixed hot-reload and restart-required changes."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test", "cortex_llm": {}}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"
            config.cortex_llm = {}

            manager = HotReloadManager(config, config_path)

            changes = {
                "name": ("test", "new_test"),
                "cortex_llm": ({}, {"type": "new"}),
            }
            result = manager._apply_changes(changes)

            assert "name" in result.hot_reloaded_fields
            assert "cortex_llm" in result.restart_required_fields
        finally:
            os.unlink(config_path)


class TestHotReloadManagerValidation:
    """Tests for HotReloadManager validation."""

    def test_validate_hertz_change(self):
        """Test validating hertz field change."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"hertz": 1}')
            config_path = f.name

        try:
            config = MagicMock()
            config.hertz = 1

            manager = HotReloadManager(config, config_path)

            assert manager._validate_change("hertz", 1, 2) is True
            assert manager._validate_change("hertz", 1, 0) is False
            assert manager._validate_change("hertz", 1, -1) is False
            assert manager._validate_change("hertz", 1, "fast") is False
        finally:
            os.unlink(config_path)

    def test_validate_non_validated_field(self):
        """Test validating field without validator."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"

            manager = HotReloadManager(config, config_path)

            assert manager._validate_change("name", "old", "new") is True
            assert manager._validate_change("unknown", "a", "b") is True
        finally:
            os.unlink(config_path)


class TestHotReloadManagerRollback:
    """Tests for HotReloadManager rollback functionality."""

    def test_rollback_stores_previous_state(self):
        """Test that rollback state is stored."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test", "hertz": 1}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"
            config.hertz = 1

            manager = HotReloadManager(config, config_path)
            manager._store_rollback_state({"name": "test", "hertz": 1})

            assert manager._rollback_state is not None
        finally:
            os.unlink(config_path)

    def test_rollback_clears_after_success(self):
        """Test that rollback state is cleared after successful reload."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"

            manager = HotReloadManager(config, config_path)
            manager._store_rollback_state({"name": "test"})
            manager._clear_rollback_state()

            assert manager._rollback_state is None
        finally:
            os.unlink(config_path)


class TestHotReloadManagerAsync:
    """Async tests for HotReloadManager."""

    @pytest.mark.asyncio
    async def test_async_on_hot_reload_callback(self):
        """Test async hot reload callback is called."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"name": "test"}')
            config_path = f.name

        try:
            config = MagicMock()
            config.name = "test"

            manager = HotReloadManager(config, config_path)

            callback_called = []

            async def async_callback(fields):
                callback_called.append(fields)

            manager.set_on_hot_reload(async_callback)

            await manager._notify_hot_reload({"name"})

            assert len(callback_called) == 1
            assert "name" in callback_called[0]
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_async_on_restart_required_callback(self):
        """Test async restart required callback is called."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"cortex_llm": {}}')
            config_path = f.name

        try:
            config = MagicMock()
            config.cortex_llm = {}

            manager = HotReloadManager(config, config_path)

            callback_called = []

            async def async_callback(fields):
                callback_called.append(fields)

            manager.set_on_restart_required(async_callback)

            await manager._notify_restart_required({"cortex_llm"})

            assert len(callback_called) == 1
            assert "cortex_llm" in callback_called[0]
        finally:
            os.unlink(config_path)

