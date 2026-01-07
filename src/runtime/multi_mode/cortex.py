import asyncio
import logging
import os
import time
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
from runtime.multi_mode.config import (
    LifecycleHookType,
    ModeSystemConfig,
    RuntimeConfig,
    load_mode_config,
)
from runtime.multi_mode.manager import ModeManager
from simulators.orchestrator import SimulatorOrchestrator


class ModeCortexRuntime:
    """
    Mode-aware cortex runtime that can dynamically switch between different
    operational modes, each with their own configuration, inputs, and actions.

    This implementation supports selective hot-reload of configuration fields,
    allowing certain fields to be updated without restarting the entire system.
    """

    mode_config: ModeSystemConfig
    mode_config_name: str
    mode_manager: ModeManager
    io_provider: IOProvider
    sleep_ticker_provider: SleepTickerProvider
    config_provider: ConfigProvider

    current_config: Optional[RuntimeConfig]
    fuser: Optional[Fuser]
    action_orchestrator: Optional[ActionOrchestrator]
    simulator_orchestrator: Optional[SimulatorOrchestrator]
    background_orchestrator: Optional[BackgroundOrchestrator]
    input_orchestrator: Optional[InputOrchestrator]

    def __init__(
        self,
        mode_config: ModeSystemConfig,
        mode_config_name: str,
        hot_reload: bool = True,
        check_interval: float = 60,
    ):
        self.mode_config = mode_config
        self.mode_config_name = mode_config_name
        self.mode_manager = ModeManager(mode_config)
        self.io_provider = IOProvider()
        self.sleep_ticker_provider = SleepTickerProvider()
        self.config_provider = ConfigProvider()

        self.hot_reload = hot_reload
        self.check_interval = check_interval
        self.config_watcher_task: Optional[asyncio.Task] = None
        self.last_modified: Optional[float] = None

        # Store raw config for diff comparison
        self._last_raw_config: Optional[Dict[str, Any]] = None

        if self.hot_reload:
            self.config_path = self.mode_manager._get_runtime_config_path()
            self.last_modified = self._get_file_mtime()
            self._last_raw_config = self._load_raw_config()
            logging.info(
                f"Hot-reload enabled for runtime config: {self.config_path} (check interval: {check_interval}s)"
            )

        self.current_config: Optional[RuntimeConfig] = None
        self.fuser: Optional[Fuser] = None
        self.action_orchestrator: Optional[ActionOrchestrator] = None
        self.simulator_orchestrator: Optional[SimulatorOrchestrator] = None
        self.background_orchestrator: Optional[BackgroundOrchestrator] = None
        self.input_orchestrator: Optional[InputOrchestrator] = None

        self.input_listener_task: Optional[asyncio.Task] = None
        self.simulator_task: Optional[asyncio.Future] = None
        self.action_task: Optional[asyncio.Future] = None
        self.background_task: Optional[asyncio.Future] = None
        self.cortex_loop_task: Optional[asyncio.Task] = None
        self.mode_transition_task: Optional[asyncio.Task] = None

        self.mode_manager.add_transition_callback(self._on_mode_transition)
        self._mode_initialized = False
        self._is_reloading = False
        self._mode_transition_event = asyncio.Event()
        self._pending_mode_transition: Optional[str] = None
        self._pending_transition_reason: Optional[str] = None

    def _load_raw_config(self) -> Optional[Dict[str, Any]]:
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json5.load(f)
        except Exception as e:
            logging.error(f"Failed to load raw config: {e}")
        return None

    async def _initialize_mode(self, mode_name: str):
        mode_config = self.mode_config.modes[mode_name]
        mode_config.load_components(self.mode_config)
        self.current_config = mode_config.to_runtime_config(self.mode_config)
        logging.info(f"Initializing mode: {mode_config.display_name}")

        self.fuser = Fuser(self.current_config)
        self.action_orchestrator = ActionOrchestrator(self.current_config)
        self.simulator_orchestrator = SimulatorOrchestrator(self.current_config)
        self.background_orchestrator = BackgroundOrchestrator(self.current_config)
        logging.info(f"Mode '{mode_name}' initialized successfully")

    async def _handle_mode_transitions(self):
        while True:
            try:
                await self._mode_transition_event.wait()
                if self._pending_mode_transition:
                    target_mode = self._pending_mode_transition
                    transition_reason = (
                        self._pending_transition_reason or "input_triggered"
                    )
                    self._pending_mode_transition = None
                    self._pending_transition_reason = None

                    logging.info(
                        f"Processing mode transition to: {target_mode} (reason: {transition_reason})"
                    )
                    success = await self.mode_manager._execute_transition(
                        target_mode, transition_reason
                    )
                    if success:
                        logging.info(
                            f"Mode transition completed successfully: {target_mode}"
                        )
                    else:
                        logging.error(f"Mode transition failed: {target_mode}")

                self._mode_transition_event.clear()

            except asyncio.CancelledError:
                logging.debug("Mode transition handler cancelled")
                break
            except Exception as e:
                logging.error(f"Error in mode transition handler: {e}")
                await asyncio.sleep(1.0)

    async def _on_mode_transition(self, from_mode: str, to_mode: str):
        logging.info(f"Handling mode transition: {from_mode} -> {to_mode}")
        try:
            self._is_reloading = True
            await self._stop_current_orchestrators()
            await self._initialize_mode(to_mode)
            await self._start_orchestrators()
            logging.info(f"Successfully transitioned to mode: {to_mode}")
        except Exception as e:
            logging.error(f"Error during mode transition {from_mode} -> {to_mode}: {e}")
            raise
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

    async def _start_orchestrators(self):
        if not self.current_config:
            raise RuntimeError("No current config available")

        self.input_orchestrator = InputOrchestrator(self.current_config.agent_inputs)
        self.input_listener_task = asyncio.create_task(self.input_orchestrator.listen())

        if self.simulator_orchestrator:
            self.simulator_task = self.simulator_orchestrator.start()
        if self.action_orchestrator:
            self.action_task = self.action_orchestrator.start()
        if self.background_orchestrator:
            self.background_task = self.background_orchestrator.start()

        self.cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

        if not self.mode_transition_task or self.mode_transition_task.done():
            self.mode_transition_task = asyncio.create_task(
                self._handle_mode_transitions()
            )

        logging.debug("Orchestrators started successfully")

    async def _cleanup_tasks(self):
        tasks_to_cancel = []

        if self.config_watcher_task and not self.config_watcher_task.done():
            tasks_to_cancel.append(self.config_watcher_task)
        if self.cortex_loop_task and not self.cortex_loop_task.done():
            tasks_to_cancel.append(self.cortex_loop_task)
        if self.mode_transition_task and not self.mode_transition_task.done():
            tasks_to_cancel.append(self.mode_transition_task)
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

    async def run(self) -> None:
        try:
            self.mode_manager.set_event_loop(asyncio.get_event_loop())

            if not self._mode_initialized:
                startup_context = {
                    "system_name": self.mode_config.name,
                    "initial_mode": self.mode_manager.current_mode_name,
                    "timestamp": asyncio.get_event_loop().time(),
                }

                startup_success = await self.mode_config.execute_global_lifecycle_hooks(
                    LifecycleHookType.ON_STARTUP, startup_context
                )
                if not startup_success:
                    logging.warning("Some global startup hooks failed")

                await self._initialize_mode(self.mode_manager.current_mode_name)
                self._mode_initialized = True

                initial_mode_config = self.mode_config.modes[
                    self.mode_manager.current_mode_name
                ]
                await initial_mode_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_STARTUP, startup_context
                )

            await self._start_orchestrators()

            if self.hot_reload and self.config_path:
                self.config_watcher_task = asyncio.create_task(
                    self._check_config_changes()
                )

            while True:
                try:
                    awaitables: List[Union[asyncio.Task, asyncio.Future]] = []
                    if self.cortex_loop_task and not self.cortex_loop_task.done():
                        awaitables.append(self.cortex_loop_task)
                    if (
                        self.mode_transition_task
                        and not self.mode_transition_task.done()
                    ):
                        awaitables.append(self.mode_transition_task)
                    if self.config_watcher_task and not self.config_watcher_task.done():
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
                    logging.debug(
                        "Tasks cancelled during mode transition, continuing..."
                    )
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error in orchestrator tasks: {e}")
                    await asyncio.sleep(1.0)

        except Exception as e:
            logging.error(f"Error in mode-aware cortex runtime: {e}")
            raise
        finally:
            shutdown_context = {
                "system_name": self.mode_config.name,
                "final_mode": self.mode_manager.current_mode_name,
                "timestamp": asyncio.get_event_loop().time(),
            }

            current_config = self.mode_config.modes.get(
                self.mode_manager.current_mode_name
            )
            if current_config:
                await current_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_SHUTDOWN, shutdown_context
                )

            await self.mode_config.execute_global_lifecycle_hooks(
                LifecycleHookType.ON_SHUTDOWN, shutdown_context
            )
            await self._cleanup_tasks()

    async def _run_cortex_loop(self) -> None:
        current_mode = self.mode_manager.current_mode_name
        logging.info(f"Starting cortex loop for mode: {current_mode}")

        try:
            while True:
                if not self.sleep_ticker_provider.skip_sleep and self.current_config:
                    await self.sleep_ticker_provider.sleep(
                        1 / self.current_config.hertz
                    )
                await asyncio.sleep(0)
                await self._tick()
                self.sleep_ticker_provider.skip_sleep = False
        except asyncio.CancelledError:
            logging.info(
                f"Cortex loop for mode '{current_mode}' cancelled, exiting gracefully"
            )
            raise
        except Exception as e:
            logging.error(
                f"Unexpected error in cortex loop for mode '{current_mode}': {e}"
            )
            raise

    async def _tick(self) -> None:
        if not self.current_config or not self.fuser or not self.action_orchestrator:
            logging.warning("Cortex not properly initialized, skipping tick")
            return

        if self._is_reloading:
            logging.debug("Skipping tick during config reload")
            return

        tick_num = self.io_provider.increment_tick()
        logging.debug(f"Processing tick #{tick_num}")

        finished_promises, _ = await self.action_orchestrator.flush_promises()
        prompt = self.fuser.fuse(self.current_config.agent_inputs, finished_promises)
        if prompt is None:
            logging.debug("No prompt to fuse")
            return

        with self.io_provider.mode_transition_input():
            last_input = self.io_provider.get_mode_transition_input()

        transition_result = await self.mode_manager.process_tick(last_input)
        if transition_result:
            new_mode, transition_reason = transition_result
            self._pending_mode_transition = new_mode
            self._pending_transition_reason = transition_reason
            self._mode_transition_event.set()
            logging.info(
                f"Scheduled mode transition to: {new_mode} (reason: {transition_reason})"
            )
            return

        output = await self.current_config.cortex_llm.ask(prompt)
        if output is None:
            logging.debug("No output from LLM")
            return

        if self._is_reloading:
            logging.debug("Skipping tick during config reload")
            return

        if self.simulator_orchestrator:
            await self.simulator_orchestrator.promise(output.actions)
        await self.action_orchestrator.promise(output.actions)

    def get_mode_info(self) -> dict:
        return self.mode_manager.get_mode_info()

    async def request_mode_change(self, target_mode: str) -> bool:
        return await self.mode_manager.request_transition(target_mode, "manual")

    def get_available_modes(self) -> dict:
        return {
            name: {
                "display_name": config.display_name,
                "description": config.description,
                "is_current": name == self.mode_manager.current_mode_name,
            }
            for name, config in self.mode_config.modes.items()
        }

    def _get_file_mtime(self) -> float:
        if self.config_path and os.path.exists(self.config_path):
            return os.path.getmtime(self.config_path)
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
                await asyncio.sleep(10)

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

                if self.current_config and hasattr(self.current_config, field):
                    old_value = getattr(self.current_config, field, None)
                    setattr(self.current_config, field, new_value)
                    logging.info(f"Hot-reloaded '{field}': {old_value} -> {new_value}")

                    if field in (
                        "system_prompt_base",
                        "system_governance",
                        "system_prompt_examples",
                    ):
                        self.fuser = Fuser(self.current_config)
                        logging.debug(f"Fuser updated due to '{field}' change")

                    if field == "hertz":
                        logging.debug(f"Hertz updated to {new_value}")

            except Exception as e:
                logging.error(f"Failed to hot-reload field '{field}': {e}")

    async def _reload_config(self) -> None:
        try:
            logging.info(f"Reloading mode configuration: {self.mode_config_name}")
            self._is_reloading = True

            current_mode = self.mode_manager.current_mode_name
            await self._stop_current_orchestrators()

            logging.info("Loading configuration from the new runtime file")
            new_mode_config = load_mode_config(
                self.mode_config_name,
                mode_source_path=self.mode_manager._get_runtime_config_path(),
            )

            self.mode_config = new_mode_config
            self.mode_manager.config = new_mode_config

            if current_mode not in new_mode_config.modes:
                logging.warning(
                    f"Current mode '{current_mode}' not found in reloaded config, "
                    f"switching to default mode '{new_mode_config.default_mode}'"
                )
                current_mode = new_mode_config.default_mode

            self.mode_manager.state.current_mode = current_mode
            self.mode_manager.state.mode_start_time = time.time()
            self.mode_manager.state.last_transition_time = time.time()
            self.mode_manager.state.transition_history.append(
                f"config_reload->{current_mode}:hot_reload"
            )

            await self._initialize_mode(current_mode)
            await self._start_orchestrators()

            logging.info(
                f"Mode configuration reloaded successfully, active mode: {current_mode}"
            )

        except Exception as e:
            logging.error(f"Failed to reload mode configuration: {e}")
            logging.error("Continuing with previous configuration")

        finally:
            self._is_reloading = False
