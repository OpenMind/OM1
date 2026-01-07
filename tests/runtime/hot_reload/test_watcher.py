"""Tests for the watcher module."""

import asyncio
import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from runtime.hot_reload.watcher import ConfigFileHandler, ConfigFileWatcher


class TestConfigFileHandler:
    """Tests for ConfigFileHandler class."""

    def test_handler_initialization(self):
        """Test handler initializes with correct parameters."""
        callback = MagicMock()
        handler = ConfigFileHandler(
            config_path="/path/to/config.json5",
            callback=callback,
            debounce_seconds=0.5,
        )
        assert handler.config_filename == "config.json5"
        assert handler.callback == callback
        assert handler.debounce_seconds == 0.5

    def test_handler_default_debounce(self):
        """Test handler uses default debounce time."""
        handler = ConfigFileHandler(
            config_path="/path/to/config.json5",
            callback=MagicMock(),
        )
        assert handler.debounce_seconds == 0.5

    def test_handler_ignores_non_target_files(self):
        """Test handler ignores modifications to other files."""
        callback = MagicMock()
        handler = ConfigFileHandler(
            config_path="/path/to/config.json5",
            callback=callback,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/other.json5"

        handler.on_modified(event)
        callback.assert_not_called()

    def test_handler_ignores_directories(self):
        """Test handler ignores directory events."""
        callback = MagicMock()
        handler = ConfigFileHandler(
            config_path="/path/to/config.json5",
            callback=callback,
        )

        event = MagicMock()
        event.is_directory = True
        event.src_path = "/path/to/config.json5"

        handler.on_modified(event)
        callback.assert_not_called()


class TestConfigFileWatcher:
    """Tests for ConfigFileWatcher class."""

    def test_watcher_initialization(self):
        """Test watcher initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert watcher.config_path == os.path.abspath(config_path)
            assert watcher._observer is None
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_start_stop(self):
        """Test watcher can be started and stopped."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            def callback(path):
                pass

            watcher = ConfigFileWatcher(config_path)
            watcher.set_callback(callback)

            result = watcher.start()
            assert result is True
            assert watcher._is_running is True
            assert watcher._observer is not None

            watcher.stop()
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_context_manager(self):
        """Test watcher works as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            def callback(path):
                pass

            with ConfigFileWatcher(config_path) as watcher:
                watcher.set_callback(callback)
                assert watcher._is_running is True

            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_set_callback(self):
        """Test setting callback function."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)

            def callback(path):
                pass

            watcher.set_callback(callback)
            assert watcher._callback == callback
        finally:
            os.unlink(config_path)

    def test_watcher_set_async_callback(self):
        """Test setting async callback function."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            loop = asyncio.new_event_loop()

            async def async_callback(path):
                pass

            watcher.set_async_callback(async_callback, loop)
            assert watcher._async_callback == async_callback
            assert watcher._event_loop == loop

            loop.close()
        finally:
            os.unlink(config_path)

    def test_watcher_thread_safety(self):
        """Test watcher is thread-safe with lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert hasattr(watcher, "_lock")
        finally:
            os.unlink(config_path)

    def test_watcher_nonexistent_file_start_fails(self):
        """Test watcher start fails for nonexistent file."""
        watcher = ConfigFileWatcher("/nonexistent/path/config.json5")
        result = watcher.start()
        assert result is False

    def test_watcher_multiple_start_stop(self):
        """Test watcher can be started and stopped multiple times."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            def callback(path):
                pass

            watcher = ConfigFileWatcher(config_path)
            watcher.set_callback(callback)

            for _ in range(3):
                result = watcher.start()
                assert result is True
                assert watcher._is_running is True
                watcher.stop()
                assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_stop_when_not_running(self):
        """Test stopping watcher when not running is safe."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            watcher.stop()
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_is_running_property(self):
        """Test is_running property."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert watcher.is_running is False

            watcher.start()
            assert watcher.is_running is True

            watcher.stop()
            assert watcher.is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_get_current_config(self):
        """Test get_current_config returns config."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"name": "test", "value": 123}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            config = watcher.get_current_config()

            assert config is not None
            assert config["name"] == "test"
            assert config["value"] == 123
        finally:
            os.unlink(config_path)

    def test_watcher_debounce_setting(self):
        """Test debounce setting."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False, mode="w") as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path, debounce_seconds=1.0)
            assert watcher.debounce_seconds == 1.0
        finally:
            os.unlink(config_path)
