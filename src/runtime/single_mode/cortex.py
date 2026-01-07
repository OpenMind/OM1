import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

import json5

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.config_provider import ConfigProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.hot_reload.diff import diff_configs
from runtime.hot_reload.strategies import (
    ReloadStrategy,
    get_field_strategy,
    validate_field,
)
from runtime.single_mode.config import RuntimeConfig, load_config
from simulators.orchestrator import SimulatorOrchestrator


class CortexRuntime:
    """
    The main entry point for the OM1 agent runtime environment.

    The CortexRuntime orchestrates communication between memory, fuser,
    actions, and manages inputs/outputs. It controls the agent's execution
    cycle and coordinates all major subsystems.

    Parameters
    ----------
    config : RuntimeConfig
        Configuration object containing all runtime settings including
        agent inputs, cortex LLM settings, and execution parameters.
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
        check_interval: float = 60,
    ):
        """
        Initialize the CortexRuntime with provided configuration.

        Parameters
        ----------
        config : RuntimeConfig
            Configuration object for the runtime.
        config_name : str
            Name of the configuration file for hot-reload functionality.
        hot_reload : bool
            Whether to enable hot-reload functionality. (default: True)
        check_interval : float
            Interval in seconds between config file checks for hot-reload. (default: 60.0)
        """
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

        self.last_modified: float = 0.0
        self.config_watcher_task: Optional[asyncio.Task] = None
        self.input_listener_task: Optional[asyncio.Task] = None
        self.simulator_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.action_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.background_task: Optional[Union[asyncio.Task, asyncio.Future]] = None
        self.cortex_loop_task: Optional[asyncio.Task] = None

        self._is_reloading = False
        self._last_raw_config: Optional[Dict[str, Any]] = None

        if self.hot_reload:
            self.config_path = self._create_runtime_config_file()
            self.last_modified = self._get_file_mtime()
            self._last_raw_config = self._load_raw_config()
            logging.info(
                f"Hot-reload enabled for runtime config: {self.config_path} (check interval: {check_interval}s)"
            )

    def _get_runtime_config_path(self) -> str:
        memory_folder_path = os.path.join(
            os.path.dirname(__file__), "../../../config", "memory"
        )
        if not os.path.exists(memory_folder_path):
            os.makedirs(memory_folder_path, mode=0o755, exist_ok=True)
        return os.path.join(memory_folder_path, ".runtime.json5")

    def _create_runtime_config_file(self) -> str:
        runtime_config_path = self._get_runtime_config_path()
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../../config",
            self.config_name + ".json5",
        )
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    raw = json5.load(f)
                tmp_path = runtime_config_path + ".tmp"
                with open(tmp_path, "w") as wf:
                    json5.dump(raw, wf, indent=2)
                os.replace(tmp_path, runtime_config_path)
                logging.debug(f"Wrote runtime config to: {runtime_config_path}")
            else:
                logging.warning(f"Config not found: {config_path}")
        except Exception as e:
            logging.error(f"Failed to create runtime config file: {e}")
        return str(runtime_config_path)

    def _load_raw_config(self) -> Optional[Dict[str, Any]]:
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json5.load(f)
        except Exception as e:
            logging.error(f"Failed to load raw config: {e}")
        return None

    async def run(self) -> None:
        """
        Start the runtime's main execution loop.

        This method initializes input listeners and begins the cortex
        processing loop, running them concurrently.

        Returns
        -------
        None
        """
        try:
            if self.hot_reload:
                self.config_watcher_task = asyncio.create_task(
                    self._check_config_changes()
                )
            await self._start_orchestrators()
            self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

            while True:
                try:
                    awaitables: List[Union[asyncio.Task, asyncio.Future]] = [
                        self.cortex_loop_task
                    ]
                    if self.hot_reload and self.config_watcher_task:
                        awaitables.append(self.config_watcher_task)
                    if self.input_listener_task and not self.input_listener_task.done():
                        awaitables.append(self.input_listener_task)
                    if self.simulator_task and not self.simulator_task.done():
                        awaitables.append(self.simulator_task)
                    if self.action_task and not self.action_task.done():
                        awaitables.append(self.action_task)
                    if self.background_task and not self.background_task.done():
                        awaitables.append(self.background_task)

                    await asyncio.gather(*awaitables)

                except asyncio.CancelledError:
                    logging.debug("Tasks cancelled during config reload, continuing...")
                    await asyncio.sleep(0.1)
                    if not self.cortex_loop_task.done():
                        continue
                    else:
                        break

                except Exception as e:
                    logging.error(f"Error in orchestrator tasks: {e}")
                    await asyncio.sleep(1.0)

        except Exception as e:
            logging.error(f"Error in cortex runtime: {e}")
            raise
        finally:
            await self._cleanup_tasks()

    def _get_file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return 0.0

    async def _check_config_changes(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                if not self.config_path or not os.path.exists(self.config_path):
                    continue

                current_mtime = self._get_file_mtime()
                if self.last_modified and current_mtime > self.last_modified:
                    logging.info(
                        f"Config file changed, analyzing changes: {self.config_path}"
                    )
                    new_raw_config = self._load_raw_config()
                    if new_raw_config is None:
                        logging.error("Failed to load new configuration")
                        continue

                    if self._last_raw_config is None:
                        await self._reload_config()
                    else:
                        await self._handle_config_changes(new_raw_config)

                    self.last_modified = current_mtime
                    self._last_raw_config = new_raw_config

            except asyncio.CancelledError:
                logging.debug("Config watcher cancelled")
                break
            except Exception as e:
                logging.error(f"Error checking config changes: {e}")
                await asyncio.sleep(5)

    async def _handle_config_changes(self, new_raw_config: Dict[str, Any]) -> None:
        diff = diff_configs(self._last_raw_config, new_raw_config)
        if not diff.has_changes:
            logging.debug("No configuration changes detected")
            return

        logging.info(f"Detected {len(diff.changes)} config change(s)")
        hot_reload_fields: Set[str] = set()
        restart_required_fields: Set[str] = set()

        for field_path, (old_val, new_val) in diff.changed_fields.items():
            strategy = get_field_strategy(field_path)
            if strategy == ReloadStrategy.IGNORE:
                continue
            elif strategy == ReloadStrategy.RESTART_REQUIRED:
                restart_required_fields.add(field_path)
                logging.info(f"  Field '{field_path}' requires restart")
            elif strategy in (ReloadStrategy.HOT_RELOAD, ReloadStrategy.VALIDATE_FIRST):
                if strategy == ReloadStrategy.VALIDATE_FIRST:
                    if not validate_field(field_path, old_val, new_val):
                        logging.error(
                            f"  Validation failed for '{field_path}', skipping"
                        )
                        continue
                hot_reload_fields.add(field_path)
                logging.info(f"  Field '{field_path}' can be hot-reloaded")

        if hot_reload_fields:
            await self._apply_hot_reload_fields(hot_reload_fields, new_raw_config)

        if restart_required_fields:
            logging.warning(
                f"Fields requiring restart: {restart_required_fields}. Performing full reload."
            )
            await self._reload_config()

    async def _apply_hot_reload_fields(
        self, fields: Set[str], new_raw_config: Dict[str, Any]
    ) -> None:
        for field in fields:
            try:
                new_value = new_raw_config.get(field)
                old_value = getattr(self.config, field, None)

                if hasattr(self.config, field):
                    setattr(self.config, field, new_value)
                    logging.info(f"Hot-reloaded '{field}': {old_value} -> {new_value}")

                    if field in (
                        "system_prompt_base",
                        "system_governance",
                        "system_prompt_examples",
                    ):
                        self.fuser = Fuser(self.config)
                        logging.debug(f"Fuser updated due to '{field}' change")

                    if field == "hertz":
                        logging.debug(f"Hertz updated to {new_value}")

            except Exception as e:
                logging.error(f"Failed to hot-reload field '{field}': {e}")

    async def _reload_config(self) -> None:
        try:
            logging.info(f"Reloading configuration: {self.config_name}")
            self._is_reloading = True

            if not self.config_name:
                logging.error("No config name available for reload")
                return

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
            self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())
            logging.info("Configuration reloaded successfully")

        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            logging.error("Continuing with previous configuration")
        finally:
            self._is_reloading = False

    async def _stop_current_orchestrators(self) -> None:
        logging.debug("Stopping current orchestrators...")
        self.sleep_ticker_provider.skip_sleep = True
        tasks_to_cancel = {}

        if self.cortex_loop_task and not self.cortex_loop_task.done():
            tasks_to_cancel["cortex_loop"] = self.cortex_loop_task
        if self.input_listener_task and not self.input_listener_task.done():
            tasks_to_cancel["input_listener"] = self.input_listener_task
        if self.simulator_task and not self.simulator_task.done():
            tasks_to_cancel["simulator"] = self.simulator_task
        if self.action_task and not self.action_task.done():
            tasks_to_cancel["action"] = self.action_task
        if self.background_task and not self.background_task.done():
            tasks_to_cancel["background"] = self.background_task

        for name, task in tasks_to_cancel.items():
            task.cancel()
            logging.debug(f"Cancelled task: {name}")

        if tasks_to_cancel:
            try:
                done, pending = await asyncio.wait(
                    tasks_to_cancel.values(),
                    timeout=1.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
                if pending:
                    logging.warning(f"Abandoning {len(pending)} unresponsive tasks")
                else:
                    logging.info(f"All {len(done)} tasks cancelled successfully!")
            except Exception as e:
                logging.warning(f"Error during task cancellation: {e}")

        self.cortex_loop_task = None
        self.input_listener_task = None
        self.simulator_task = None
        self.action_task = None
        self.background_task = None

    async def _start_orchestrators(self) -> None:
        logging.debug("Starting orchestrators...")
        input_orchestrator = InputOrchestrator(self.config.agent_inputs)
        self.input_listener_task = asyncio.create_task(input_orchestrator.listen())

        if self.simulator_orchestrator:
            self.simulator_task = self.simulator_orchestrator.start()
        if self.action_orchestrator:
            self.action_task = self.action_orchestrator.start()
        if self.background_orchestrator:
            self.background_task = self.background_orchestrator.start()

        logging.debug("Orchestrators started successfully")

    async def _cleanup_tasks(self) -> None:
        tasks_to_cancel = []
        if self.config_watcher_task and not self.config_watcher_task.done():
            tasks_to_cancel.append(self.config_watcher_task)
        if self.cortex_loop_task and not self.cortex_loop_task.done():
            tasks_to_cancel.append(self.cortex_loop_task)
        if self.input_listener_task and not self.input_listener_task.done():
            tasks_to_cancel.append(self.input_listener_task)
        if self.simulator_task and not self.simulator_task.done():
            tasks_to_cancel.append(self.simulator_task)
        if self.action_task and not self.action_task.done():
            tasks_to_cancel.append(self.action_task)
        if self.background_task and not self.background_task.done():
            tasks_to_cancel.append(self.background_task)

        for task in tasks_to_cancel:
            task.cancel()

        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Error during final cleanup: {e}")

        self.config_provider.stop()
        logging.debug("Tasks cleaned up successfully")

    async def _run_cortex_loop(self) -> None:
        try:
            while True:
                if not self.sleep_ticker_provider.skip_sleep:
                    await self.sleep_ticker_provider.sleep(1 / self.config.hertz)
                await asyncio.sleep(0)
                await self._tick()
                self.sleep_ticker_provider.skip_sleep = False
        except asyncio.CancelledError:
            logging.info("Cortex loop cancelled, exiting gracefully")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in cortex loop: {e}")
            raise

    async def _tick(self) -> None:
        try:
            if self._is_reloading:
                logging.debug("Skipping tick during config reload")
                return

            tick_num = self.io_provider.increment_tick()
            logging.debug(f"Processing tick #{tick_num}")

            finished_promises, _ = await self.action_orchestrator.flush_promises()
            prompt = self.fuser.fuse(self.config.agent_inputs, finished_promises)
            if prompt is None:
                logging.debug("No prompt to fuse")
                return

            output = await self.config.cortex_llm.ask(prompt)
            if output is None:
                logging.debug("No output from LLM")
                return

            await self.simulator_orchestrator.promise(output.actions)
            await self.action_orchestrator.promise(output.actions)
        except Exception as error:
            logging.error(f"Error in cortex tick: {error}")
