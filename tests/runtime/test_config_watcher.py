import asyncio
import tempfile
from pathlib import Path

import pytest
from watchdog.events import FileModifiedEvent

from utils.config_watcher import ConfigFileWatcher


class TestConfigWatcherAsyncCallback:
    @pytest.mark.asyncio
    async def test_run_coroutine_threadsafe_execution(self):
        """Test line 80 - asyncio.run_coroutine_threadsafe"""
        callback_executed = asyncio.Event()

        async def async_callback(path: Path):
            callback_executed.set()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        watcher = None
        try:
            loop = asyncio.get_running_loop()
            watcher = ConfigFileWatcher(
                config_path=config_path,
                on_change_callback=async_callback,
                debounce_seconds=0.1,
            )
            watcher.start(event_loop=loop)
            await asyncio.sleep(0.05)

            if watcher._handler:
                event = FileModifiedEvent(str(config_path))
                watcher._handler.on_modified(event)

            await asyncio.wait_for(callback_executed.wait(), timeout=1.0)
            assert callback_executed.is_set()
        finally:
            if watcher is not None:
                watcher.stop()
            config_path.unlink(missing_ok=True)
