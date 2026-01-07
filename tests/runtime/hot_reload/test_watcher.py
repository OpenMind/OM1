"""Tests for the watcher module."""

import asyncio
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

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
        assert handler.config_path == "/path/to/config.json5"
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
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert watcher.config_path == config_path
            assert watcher._observer is None
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_start_stop(self):
        """Test watcher can be started and stopped."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            callback = MagicMock()
            watcher = ConfigFileWatcher(config_path)
            watcher.set_callback(callback)

            watcher.start()
            assert watcher._is_running is True
            assert watcher._observer is not None

            watcher.stop()
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_context_manager(self):
        """Test watcher works as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            callback = MagicMock()
            with ConfigFileWatcher(config_path) as watcher:
                watcher.set_callback(callback)
                watcher.start()
                assert watcher._is_running is True

            assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_set_callback(self):
        """Test setting callback function."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            callback = MagicMock()
            watcher.set_callback(callback)
            assert watcher._callback == callback
        finally:
            os.unlink(config_path)

    def test_watcher_set_async_callback(self):
        """Test setting async callback function."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)

            async def async_callback():
                pass

            watcher.set_async_callback(async_callback)
            assert watcher._async_callback == async_callback
        finally:
            os.unlink(config_path)

    def test_watcher_thread_safety(self):
        """Test watcher is thread-safe with lock."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            assert hasattr(watcher, "_lock")
        finally:
            os.unlink(config_path)

    def test_watcher_nonexistent_file(self):
        """Test watcher handles nonexistent file gracefully."""
        watcher = ConfigFileWatcher("/nonexistent/path/config.json5")
        assert watcher.config_path == "/nonexistent/path/config.json5"

    def test_watcher_multiple_start_stop(self):
        """Test watcher can be started and stopped multiple times."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            callback = MagicMock()
            watcher = ConfigFileWatcher(config_path)
            watcher.set_callback(callback)

            for _ in range(3):
                watcher.start()
                assert watcher._is_running is True
                watcher.stop()
                assert watcher._is_running is False
        finally:
            os.unlink(config_path)

    def test_watcher_stop_when_not_running(self):
        """Test stopping watcher when not running is safe."""
        with tempfile.NamedTemporaryFile(suffix=".json5", delete=False) as f:
            f.write(b'{"test": true}')
            config_path = f.name

        try:
            watcher = ConfigFileWatcher(config_path)
            watcher.stop()
            assert watcher._is_running is False
        finally:
            os.unlink(config_path)


class TestConfigFileWatcherIntegration:
    """Integration tests for ConfigFileWatcher."""

    @pytest.mark.slow
    def test_watcher_detects_file_modification(self):
        """Test watcher detects file modifications."""
        with tempfile.NamedTemporaryFile(
            suffix=".json5", delete=False, mode="w"
        ) as f:
            f.write('{"test": true}')
            config_path = f.name

        try:
            callback_called = []
            callback = lambda: callback_called.append(True)

            watcher = ConfigFileWatcher(config_path, debounce_seconds=0.1)
            watcher.set_callback(callback)
            watcher.start()

            time.sleep(0.2)

            with open(config_path, "w") as f:
                f.write('{"test": false}')

            time.sleep(0.5)

            watcher.stop()

            assert len(callback_called) >= 1
        finally:
            os.unlink(config_path)

    @pytest.mark.slow
    def test_watcher_debounces_rapid_changes(self):
        """Test watcher debounces rapid consecutive changes."""
        with tempfile.NamedTemporaryFile(
            suffix=".json5", delete=False, mode="w"
        ) as f:
            f.write('{"count": 0}')
            config_path = f.name

        try:
            callback_count = []
            callback = lambda: callback_count.append(1)

            watcher = ConfigFileWatcher(config_path, debounce_seconds=0.3)
            watcher.set_callback(callback)
            watcher.start()

            time.sleep(0.1)

            for i in range(5):
                with open(config_path, "w") as f:
                    f.write(f'{{"count": {i}}}')
                time.sleep(0.05)

            time.sleep(0.5)

            watcher.stop()

            assert len(callback_count) <= 2
        finally:
            os.unlink(config_path)

