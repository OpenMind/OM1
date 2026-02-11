"""Cortex Runtime Module."""

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
from runtime.single_mode.hot_reload import HotReloadManager, ReloadStrategy
from simulators.orchestrator import SimulatorOrchestrator
from utils.config_watcher import ConfigFileWatcher


class CortexRuntime:
    """
    The main entry point for the OM1 agent runtime environment.

    The CortexRuntime orchestrates communication between memory, fuser,
    actions, and manages inputs/outputs. It controls the agent's execution
    cycle and coordinates all major subsystems.
    """

    config: RuntimeConfig
    fuser: Fuser
    action_orchestrator: ActionOrchestrator
    simulator_orchestrator: SimulatorOrchestrator
    background_orchestrator: BackgroundOrchestrator
    sleep_ticker_provider: SleepTickerProvider
    io_provider: IOProvider
    config_provider: ConfigProvider

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

    async def _on_config_file_changed(self, path: Path):
        logging.info("Config file changed: %s", path)
        await self._reload_config()

    async def _reload_config(self, new_config: Optional[RuntimeConfig] = None):
        if not self.config_name:
            return

        if new_config is None:
            try:
                new_config = load_config(
                    self.config_name, config_source_path=self.config_path
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Failed to load new config: %s", e)
                return

        old_config_dict = self.config.to_dict()
        new_config_dict = new_config.to_dict()

        detected_changes = self.hot_reload_manager.detect_changes(
            old_config_dict, new_config_dict
        )
        if not detected_changes:
            logging.info("Config file changed, no registered fields modified.")
            return

        categorized_changes = self.hot_reload_manager.categorize_changes(
            detected_changes
        )

        restart_required_changes = categorized_changes[ReloadStrategy.RESTART_REQUIRED]
        validate_first_changes = categorized_changes[ReloadStrategy.VALIDATE_FIRST]
        hot_reload_changes = categorized_changes[ReloadStrategy.HOT_RELOAD]

        if restart_required_changes:
            logging.warning("Restart required for changes:")
            for change in restart_required_changes:
                logging.warning(
                    "  - %s: %s -> %s",
                    change.field_path,
                    change.old_value,
                    change.new_value,
                )

        if validate_first_changes:
            logging.info("Validating changes:")
            for change in validate_first_changes:
                logging.info(
                    "  - %s: %s -> %s",
                    change.field_path,
                    change.old_value,
                    change.new_value,
                )

        if hot_reload_changes:
            logging.info("Hot-reloading changes:")
            for change in hot_reload_changes:
                logging.info(
                    "  - %s: %s -> %s",
                    change.field_path,
                    change.old_value,
                    change.new_value,
                )

        self._is_reloading = True

        if restart_required_changes:
            logging.info("Restart required. Stopping orchestrators.")
            await self._stop_current_orchestrators()
            self.config = new_config
            logging.warning("Full restart recommended.")
            self._is_reloading = False
            return

        validation_results = self.hot_reload_manager.validate_changes(
            validate_first_changes
        )
        if not all(validation_results.values()):
            invalid_fields = [f for f, v in validation_results.items() if not v]
            logging.error("Validation failed: %s. Aborting hot-reload.", invalid_fields)
            self._is_reloading = False
            return

        for change in validate_first_changes + hot_reload_changes:
            self._update_nested_config(
                self.config.__dict__, change.field_path, change.new_value
            )

        if any(
            "system_prompt" in change.field_path
            for change in (validate_first_changes + hot_reload_changes)
        ):
            self.fuser = Fuser(self.config)
            logging.info("Fuser updated.")

        if any(
            "cortex_llm.config" in change.field_path
            for change in (validate_first_changes + hot_reload_changes)
        ):
            # Assumes ActionOrchestrator has update_config method.
            self.action_orchestrator.update_config(self.config)
            logging.info("ActionOrchestrator LLM config updated.")

        for change in validate_first_changes + hot_reload_changes:
            self.hot_reload_manager.track_change(change)

        logging.info("Selective hot-reload completed.")
        self._is_reloading = False

    def _update_nested_config(self, config_obj, path: str, new_value):
        keys = path.split(".")
        current = config_obj
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict):
                current = current[key]
            else:
                logging.error("Could not navigate config: %s", path)
                return
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, new_value)
        elif isinstance(current, dict):
            current[final_key] = new_value
        else:
            logging.error("Could not set config key: %s in %s", final_key, current)

    async def _start_orchestrators(self):
        self.input_listener_task = await self._start_input_listeners()
        self.simulator_task = self.simulator_orchestrator.start()
        self.action_task = self.action_orchestrator.start()
        self.background_task = self.background_orchestrator.start()

    async def run(self) -> None:
        """
        Start the runtime's main execution loop.

        This method initializes input listeners and begins the cortex
        processing loop, running them concurrently.

        Returns
        -------
        None
        """
        if self.hot_reload:
            self.config_watcher = ConfigFileWatcher(
                config_path=Path(self.config_path),
                callback=self._on_config_file_changed,
            )
            self.config_watcher.start(event_loop=asyncio.get_running_loop())
            logging.info("Hot-reload via watchdog started.")

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

    def _get_runtime_config_path(self) -> str:
        base = Path(__file__).parent / "../../../config/memory"
        base.mkdir(parents=True, exist_ok=True)
        return str(base / ".runtime.json5")

    def _create_runtime_config_file(self) -> str:
        runtime_path = self._get_runtime_config_path()
        src = Path(__file__).parent / "../../../config" / f"{self.config_name}.json5"
        if src.exists():
            with open(src, encoding="utf-8") as f:
                data = json5.load(f)
            with open(runtime_path, "w", encoding="utf-8") as wf:
                json5.dump(data, wf, indent=2)
        return runtime_path
