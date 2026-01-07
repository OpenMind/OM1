import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import json5

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.config_provider import ConfigProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.single_mode.config import RuntimeConfig, load_config
from runtime.single_mode.hot_reload import HotReloadManager
from simulators.orchestrator import SimulatorOrchestrator
from utils.config_watcher import ConfigFileWatcher


class CortexRuntime:
    """
    Main runtime controller for single-mode Cortex execution.

    This class coordinates input listeners, orchestrators, and the main
    execution loop. It also optionally supports hot-reloading of runtime
    configuration via a watchdog-based file watcher.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        config_name: str,
        hot_reload: bool = True,
        check_interval: float = 60.0,
    ):
        self.config = config
        self.config_name = config_name
        self.hot_reload = hot_reload
        self.check_interval = check_interval

        self.fuser = Fuser(config)
        self.action_orchestrator = ActionOrchestrator(config)
        self.simulator_orchestrator = SimulatorOrchestrator(config)
        self.background_orchestrator = BackgroundOrchestrator(config)
        self.sleep_ticker_provider = SleepTickerProvider()
        self.io_provider = IOProvider()
        self.config_provider = ConfigProvider()

        self.hot_reload_manager = HotReloadManager()

        self.config_path = self._create_runtime_config_file()
        self.last_modified = self._get_file_mtime() if hot_reload else 0.0

        self.config_watcher: Optional[ConfigFileWatcher] = None
        self.config_watcher_task: Optional[asyncio.Task] = None

        self.input_listener_task = None
        self.simulator_task = None
        self.action_task = None
        self.background_task = None
        self.cortex_loop_task = None

        self._is_reloading = False

    # ------------------------------------------------------------------
    # Legacy helpers (TEST DEPENDS ON THESE)
    # ------------------------------------------------------------------

    def _get_file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return 0.0

    async def _check_config_changes(self):
        while True:
            await asyncio.sleep(self.check_interval)

            current_mtime = self._get_file_mtime()
            if current_mtime <= self.last_modified:
                continue

            self.last_modified = current_mtime
            await self._reload_config()

    async def _start_input_listeners(self) -> asyncio.Task:
        orchestrator = InputOrchestrator(self.config.agent_inputs)
        return asyncio.create_task(orchestrator.listen())

    async def _stop_current_orchestrators(self):
        tasks = {
            "input": self.input_listener_task,
            "simulator": self.simulator_task,
            "action": self.action_task,
            "background": self.background_task,
        }

        for task in tasks.values():
            if task and not task.done():
                task.cancel()

        if tasks:
            await asyncio.wait(
                [t for t in tasks.values() if t],
                timeout=1.0,
                return_when=asyncio.ALL_COMPLETED,
            )

        self.input_listener_task = None
        self.simulator_task = None
        self.action_task = None
        self.background_task = None

    async def _cleanup_tasks(self):
        tasks = []

        if self.config_watcher_task and not self.config_watcher_task.done():
            tasks.append(self.config_watcher_task)

        if self.cortex_loop_task and not self.cortex_loop_task.done():
            tasks.append(self.cortex_loop_task)

        if tasks:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        self.config_provider.stop()

    # ------------------------------------------------------------------
    # Reload logic (TESTED HEAVILY)
    # ------------------------------------------------------------------

    async def _reload_config(self, new_config: Optional[RuntimeConfig] = None):
        if not self.config_name:
            return

        try:
            if new_config is None:
                new_config = load_config(
                    self.config_name, config_source_path=self.config_path
                )

            await self._stop_current_orchestrators()

            self.config = new_config
            self.fuser = Fuser(new_config)
            self.action_orchestrator = ActionOrchestrator(new_config)
            self.simulator_orchestrator = SimulatorOrchestrator(new_config)
            self.background_orchestrator = BackgroundOrchestrator(new_config)

            await self._start_orchestrators()

        except Exception as e:
            logging.error(f"Failed to reload config: {e}")

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    async def _start_orchestrators(self):
        self.input_listener_task = await self._start_input_listeners()
        self.simulator_task = self.simulator_orchestrator.start()
        self.action_task = self.action_orchestrator.start()
        self.background_task = self.background_orchestrator.start()

    async def run(self):
        """
        Start the Cortex runtime.

        This method initializes orchestrators, starts the main Cortex loop,
        and (if enabled) launches the configuration watcher for hot reload.
        It blocks until the runtime is stopped or cancelled.
        """
        if self.hot_reload:
            self.config_watcher_task = asyncio.create_task(self._check_config_changes())

        await self._start_orchestrators()
        self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

        await self.cortex_loop_task

    async def _run_cortex_loop(self):
        interval = 1 / self.config.hertz

        while True:
            await self.sleep_ticker_provider.sleep(interval)
            await self._tick()

    async def _tick(self):
        if self._is_reloading:
            return

        finished, _ = await self.action_orchestrator.flush_promises()
        prompt = self.fuser.fuse(self.config.agent_inputs, finished)
        if not prompt:
            return

        output = await self.config.cortex_llm.ask(prompt)
        if not output:
            return

        await self.simulator_orchestrator.promise(output.actions)
        await self.action_orchestrator.promise(output.actions)

    # ------------------------------------------------------------------

    def _get_runtime_config_path(self) -> str:
        base = Path(__file__).parent / "../../../config/memory"
        base.mkdir(parents=True, exist_ok=True)
        return str(base / ".runtime.json5")

    def _create_runtime_config_file(self) -> str:
        runtime_path = self._get_runtime_config_path()
        src = Path(__file__).parent / "../../../config" / f"{self.config_name}.json5"

        if src.exists():
            with open(src) as f:
                data = json5.load(f)
            with open(runtime_path, "w") as wf:
                json5.dump(data, wf, indent=2)

        return runtime_path
