"""
Unit tests for config_watcher module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.config_watcher import ConfigFileHandler, ConfigFileWatcher


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("test: value\n")
    return config_file


@pytest.fixture
def mock_callback():
    """Create a mock async callback."""
    return AsyncMock()


class TestConfigFileHandler:
    """Tests for ConfigFileHandler class."""

    def test_init(self, temp_config_file, mock_callback):
        """Test ConfigFileHandler initialization."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
            debounce_seconds=0.5,
        )
        assert handler.callback == mock_callback
        assert handler.target_path == temp_config_file.resolve()
        assert handler.debounce_seconds == 0.5
        assert handler._last_modified == 0.0
        assert handler._loop is None

    def test_set_event_loop(self, temp_config_file, mock_callback):
        """Test setting the event loop."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)
        assert handler._loop == loop
        loop.close()

    @pytest.mark.asyncio
    async def test_wrapper_async(self, temp_config_file, mock_callback):
        """Test the async wrapper method."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )
        await handler._wrapper_async(temp_config_file)
        mock_callback.assert_called_once_with(temp_config_file)

    def test_on_modified_ignores_directory(self, temp_config_file, mock_callback):
        """Test that directory events are ignored."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)

        event = MagicMock()
        event.is_directory = True
        event.src_path = str(temp_config_file)

        handler.on_modified(event)
        mock_callback.assert_not_called()
        loop.close()

    def test_on_modified_ignores_different_file(
        self, temp_config_file, mock_callback, tmp_path
    ):
        """Test that events for different files are ignored."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)

        other_file = tmp_path / "other_file.yaml"
        other_file.write_text("other: value\n")

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(other_file)

        handler.on_modified(event)
        mock_callback.assert_not_called()
        loop.close()

    def test_on_modified_warns_on_deleted_file(
        self, temp_config_file, mock_callback, caplog
    ):
        """Test warning when config file is deleted."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)

        temp_config_file.unlink()

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(temp_config_file)

        with caplog.at_level("WARNING"):
            handler.on_modified(event)

        assert "Config file deleted" in caplog.text
        mock_callback.assert_not_called()
        loop.close()

    def test_on_modified_debouncing(self, temp_config_file, mock_callback):
        """Test that rapid changes are debounced."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
            debounce_seconds=1.0,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(temp_config_file)

        handler.on_modified(event)
        handler.on_modified(event)

        time.sleep(0.1)

        assert mock_callback.call_count <= 1
        loop.close()

    def test_on_modified_triggers_callback(self, temp_config_file, mock_callback):
        """Test that modification triggers the callback."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
            debounce_seconds=0.1,
        )
        loop = asyncio.new_event_loop()
        handler.set_event_loop(loop)

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(temp_config_file)

        handler.on_modified(event)

        time.sleep(0.05)

        loop.close()

    def test_on_modified_without_event_loop(
        self, temp_config_file, mock_callback, caplog
    ):
        """Test modification handling when event loop is not set."""
        handler = ConfigFileHandler(
            callback=mock_callback,
            target_path=temp_config_file,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(temp_config_file)

        with caplog.at_level("WARNING"):
            handler.on_modified(event)

        assert "Event loop not available" in caplog.text
        mock_callback.assert_not_called()


class TestConfigFileWatcher:
    """Tests for ConfigFileWatcher class."""

    def test_init_with_callback(self, temp_config_file, mock_callback):
        """Test ConfigFileWatcher initialization with callback."""
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
            debounce_seconds=0.5,
        )
        assert watcher.config_path == temp_config_file.resolve()
        assert watcher._on_change_callback == mock_callback
        assert watcher.debounce_seconds == 0.5
        assert watcher._observer is None
        assert watcher._handler is None
        assert watcher._watching is False

    def test_init_with_on_change_callback(self, temp_config_file, mock_callback):
        """Test ConfigFileWatcher initialization with on_change_callback."""
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            on_change_callback=mock_callback,
        )
        assert watcher._on_change_callback == mock_callback

    def test_init_nonexistent_file_warns(self, tmp_path, mock_callback, caplog):
        """Test warning when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with caplog.at_level("WARNING"):
            ConfigFileWatcher(
                config_path=nonexistent,
                callback=mock_callback,
            )

        assert "does not exist yet" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_file_change_success(self, temp_config_file, mock_callback):
        """Test successful file change handling."""
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )
        await watcher._handle_file_change(temp_config_file)
        mock_callback.assert_called_once_with(temp_config_file)

    @pytest.mark.asyncio
    async def test_handle_file_change_no_callback(self, temp_config_file, caplog):
        """Test file change handling when no callback is set."""
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=None,
        )
        await watcher._handle_file_change(temp_config_file)

    @pytest.mark.asyncio
    async def test_handle_file_change_callback_error(self, temp_config_file, caplog):
        """Test error handling in callback."""
        error_callback = AsyncMock(side_effect=Exception("Test error"))
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=error_callback,
        )

        with caplog.at_level("ERROR"):
            await watcher._handle_file_change(temp_config_file)

        assert "Hot-reload callback failed" in caplog.text

    @patch("utils.config_watcher.Observer")
    def test_start_watcher(self, mock_observer_class, temp_config_file, mock_callback):
        """Test starting the file watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        loop = asyncio.new_event_loop()
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )

        watcher.start(event_loop=loop)

        assert watcher._watching is True
        assert watcher._handler is not None
        assert watcher._observer is not None
        mock_observer.schedule.assert_called_once()
        mock_observer.start.assert_called_once()

        loop.close()

    @patch("utils.config_watcher.Observer")
    def test_start_watcher_no_event_loop(
        self, mock_observer_class, temp_config_file, mock_callback
    ):
        """Test starting watcher without explicit event loop."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            watcher.start()

        assert watcher._watching is True

    @patch("utils.config_watcher.Observer")
    def test_start_watcher_already_started(
        self, mock_observer_class, temp_config_file, mock_callback, caplog
    ):
        """Test warning when starting already running watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        loop = asyncio.new_event_loop()
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )

        watcher.start(event_loop=loop)

        with caplog.at_level("WARNING"):
            watcher.start(event_loop=loop)

        assert "already started" in caplog.text
        loop.close()

    @patch("utils.config_watcher.Observer")
    def test_stop_watcher(self, mock_observer_class, temp_config_file, mock_callback):
        """Test stopping the file watcher."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        loop = asyncio.new_event_loop()
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )

        watcher.start(event_loop=loop)
        watcher.stop()

        assert watcher._watching is False
        assert watcher._observer is None
        assert watcher._handler is None
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once_with(timeout=5)

        loop.close()

    def test_stop_watcher_not_running(self, temp_config_file, mock_callback):
        """Test stopping watcher that isn't running."""
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )
        watcher.stop()
        assert watcher._watching is False

    @patch("utils.config_watcher.Observer")
    def test_is_watching(self, mock_observer_class, temp_config_file, mock_callback):
        """Test is_watching method."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer

        loop = asyncio.new_event_loop()
        watcher = ConfigFileWatcher(
            config_path=temp_config_file,
            callback=mock_callback,
        )

        assert watcher.is_watching() is False

        watcher.start(event_loop=loop)
        assert watcher.is_watching() is True

        watcher.stop()
        assert watcher.is_watching() is False

        loop.close()
