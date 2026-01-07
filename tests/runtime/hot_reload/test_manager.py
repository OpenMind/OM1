"""Tests for the manager module."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from runtime.hot_reload.manager import HotReloadManager, ReloadResult


class TestReloadResult:
    """Tests for ReloadResult dataclass."""

    def test_default_result(self):
        """Test default ReloadResult values."""
        result = ReloadResult()
        assert result.success is True
        assert result.hot_reloaded_fields == set()
        assert result.requires_restart is False
        assert result.restart_fields == set()
        assert result.errors == []

    def test_result_with_hot_reloaded_fields(self):
        """Test ReloadResult with hot-reloaded fields."""
        result = ReloadResult(
            success=True,
            hot_reloaded_fields={"hertz", "name"},
        )
        assert result.success is True
        assert "hertz" in result.hot_reloaded_fields

    def test_result_with_restart_required(self):
        """Test ReloadResult with restart required."""
        result = ReloadResult(
            success=True,
            requires_restart=True,
            restart_fields={"cortex_llm"},
        )
        assert result.requires_restart is True
        assert "cortex_llm" in result.restart_fields

    def test_result_with_errors(self):
        """Test ReloadResult with errors."""
        result = ReloadResult(
            success=False,
            errors=["Failed to validate field"],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestHotReloadManager:
    """Tests for HotReloadManager class."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test", "hertz": 1}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            assert manager.config_path == os.path.abspath(config_path)
            assert manager.is_running is False
        finally:
            os.unlink(config_path)

    def test_manager_with_callbacks(self):
        """Test manager with callbacks."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            hot_reload_cb = MagicMock(return_value=True)
            restart_cb = MagicMock()

            manager = HotReloadManager(
                config_path,
                on_hot_reload=hot_reload_cb,
                on_restart_required=restart_cb,
            )
            assert manager.on_hot_reload == hot_reload_cb
            assert manager.on_restart_required == restart_cb
        finally:
            os.unlink(config_path)

    def test_manager_debounce_setting(self):
        """Test manager debounce setting."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path, debounce_seconds=1.0)
            assert manager.debounce_seconds == 1.0
        finally:
            os.unlink(config_path)

    def test_manager_max_history(self):
        """Test manager max history setting."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path, max_history=10)
            assert manager.max_history == 10
        finally:
            os.unlink(config_path)

    def test_manager_context_manager(self):
        """Test manager works as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            with HotReloadManager(config_path) as manager:
                assert manager.config_path == os.path.abspath(config_path)
        finally:
            os.unlink(config_path)

    def test_manager_has_lock(self):
        """Test manager has thread lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            assert hasattr(manager, "_lock")
        finally:
            os.unlink(config_path)

    def test_get_stats(self):
        """Test get_stats returns expected structure."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            stats = manager.get_stats()

            assert "is_running" in stats
            assert "config_path" in stats
            assert "history_count" in stats
            assert stats["is_running"] is False
        finally:
            os.unlink(config_path)

    def test_get_current_config_before_start(self):
        """Test get_current_config returns None before start."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            config = manager.get_current_config()
            assert config is None
        finally:
            os.unlink(config_path)

    def test_get_pending_restart_fields(self):
        """Test get_pending_restart_fields returns empty set initially."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            fields = manager.get_pending_restart_fields()
            assert fields == set()
        finally:
            os.unlink(config_path)

    def test_get_change_history(self):
        """Test get_change_history returns empty list initially."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            history = manager.get_change_history()
            assert history == []
        finally:
            os.unlink(config_path)


class TestHotReloadManagerAsync:
    """Async tests for HotReloadManager."""

    @pytest.mark.asyncio
    async def test_manager_start_stop(self):
        """Test manager start and stop."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test", "hertz": 1}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)

            started = await manager.start()
            assert started is True
            assert manager.is_running is True

            manager.stop()
            assert manager.is_running is False
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_manager_get_config_after_start(self):
        """Test get_current_config returns config after start."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test", "hertz": 1}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            await manager.start()

            config = manager.get_current_config()
            assert config is not None
            assert config["name"] == "test"
            assert config["hertz"] == 1

            manager.stop()
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_manager_async_context_manager(self):
        """Test async context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test"}')
            config_path = f.name

        try:
            async with HotReloadManager(config_path) as manager:
                assert manager.is_running is True

            assert manager.is_running is False
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_rollback_to_original(self):
        """Test rollback to original config."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test", "hertz": 1}')
            config_path = f.name

        try:
            manager = HotReloadManager(config_path)
            await manager.start()

            success = manager.rollback_to_original()
            assert success is True

            config = manager.get_current_config()
            assert config["name"] == "test"

            manager.stop()
        finally:
            os.unlink(config_path)
