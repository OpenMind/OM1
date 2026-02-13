import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.config_provider import ConfigProvider
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.config import (
    LifecycleHookType,
    ModeSystemConfig,
    RuntimeConfig,
    load_mode_config,
    mode_config_to_dict,
)
from runtime.hot_reload import HotReloadManager, ReloadStrategy
from runtime.manager import ModeManager
from simulators.orchestrator import SimulatorOrchestrator
from utils.config_watcher import ConfigFileWatcher


class ModeCortexRuntime:
    """
    Mode-aware cortex runtime that can dynamically switch between different
    operational modes, each with their own configuration, inputs, and actions.
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

        self.hot_reload_manager = HotReloadManager()
        self.hot_reload = hot_reload

        self.config_watcher: Optional[ConfigFileWatcher] = None
        if self.hot_reload:
            self.config_path = self.mode_manager.runtime_config_path
            self.config_watcher = ConfigFileWatcher(
                config_path=Path(self.config_path),
                on_change_callback=self._reload_config,
                debounce_seconds=0.5,
            )
            logging.info(
                f"Hot-reload enabled with watchdog for: {self.config_path} (debounce: 0.5s)"
            )
        else:
            self.config_path = None

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

    async def _initialize_mode(self, mode_name: str):
        mode_config = self.mode_config.modes[mode_name]
        mode_config.load_components(self.mode_config)

        self.current_config = mode_config.to_runtime_config(self.mode_config)

        logging.info(f"Initializing mode: {mode_config.display_name}")

        self.mode_manager.state.user_context.clear()

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

        if self.background_orchestrator:
            self.background_orchestrator.stop()
        if self.simulator_orchestrator:
            self.simulator_orchestrator.stop()
        if self.action_orchestrator:
            self.action_orchestrator.stop()

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
                    pending_names = [
                        name
                        for name, task in tasks_to_cancel.items()
                        if task in pending
                    ]
                    logging.warning(
                        f"Abandoning {len(pending)} unresponsive tasks: {pending_names}"
                    )
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

        self.sleep_ticker_provider.skip_sleep = False

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

        if self.config_watcher:
            self.config_watcher.stop()
            self.config_watcher = None

        self.config_provider.stop()
        logging.debug("Tasks cleaned up successfully")

    async def run(self) -> None:
        """Start the mode-aware runtime's main execution loop."""
        try:
            self.mode_manager.set_event_loop(asyncio.get_event_loop())

            if not self._mode_initialized:
                startup_context = {
                    "system_name": self.mode_config.name,
                    "initial_mode": self.mode_manager.current_mode_name,
                    "timestamp": asyncio.get_event_loop().time(),
                }

                await self.mode_config.execute_global_lifecycle_hooks(
                    LifecycleHookType.ON_STARTUP, startup_context
                )

                await self._initialize_mode(self.mode_manager.current_mode_name)
                self._mode_initialized = True

                initial_mode_config = self.mode_config.modes[
                    self.mode_manager.current_mode_name
                ]
                await initial_mode_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_STARTUP, startup_context
                )

            await self._start_orchestrators()

            if self.hot_reload and self.config_watcher:
                loop = asyncio.get_event_loop()
                self.config_watcher.start(loop)

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
                skip_status = self.sleep_ticker_provider.skip_sleep
                sleep_duration = (
                    1 / self.current_config.hertz if self.current_config else 1
                )
                if not skip_status and self.current_config:
                    await self.sleep_ticker_provider.sleep(sleep_duration)

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
        """Get information about current mode and available transitions."""
        return self.mode_manager.get_mode_info()

    async def request_mode_change(self, target_mode: str) -> bool:
        """Request manual mode change."""
        return await self.mode_manager.request_transition(target_mode, "manual")

    def get_available_modes(self) -> dict:
        """Get information about all available modes."""
        return {
            name: {
                "display_name": config.display_name,
                "description": config.description,
                "is_current": name == self.mode_manager.current_mode_name,
            }
            for name, config in self.mode_config.modes.items()
        }

    async def _reload_config(self, _: Optional[str] = None) -> None:
        if self._is_reloading:
            logging.debug("Reload already in progress, skipping")
            return

        try:
            logging.info(f"Config file changed, triggering reload: {self.config_path}")
            self._is_reloading = True

            current_mode = self.mode_manager.current_mode_name

            new_mode_config = load_mode_config(
                self.mode_config_name,
                mode_source_path=self.config_path,
            )

            current_mode_config = self.mode_config.modes.get(current_mode)
            new_mode_config_entry = new_mode_config.modes.get(current_mode)

            if not current_mode_config or not new_mode_config_entry:
                logging.warning(
                    f"Mode '{current_mode}' not found in reloaded config, doing full restart"
                )
                await self._full_restart_reload(new_mode_config, current_mode)
                return

            old_dict = mode_config_to_dict(self.mode_config)
            new_dict = mode_config_to_dict(new_mode_config)

            changes = self.hot_reload_manager.detect_changes(old_dict, new_dict)

            if not changes:
                logging.info("Config file changed, but no registered fields modified.")
                self.mode_config = new_mode_config
                self.mode_manager.config = new_mode_config
                return

            categorized = self.hot_reload_manager.categorize_changes(changes)

            if categorized[ReloadStrategy.RESTART_REQUIRED]:
                logging.warning(
                    "Restart required for changes: %s",
                    [
                        c.field_path
                        for c in categorized[ReloadStrategy.RESTART_REQUIRED]
                    ],
                )
                await self._full_restart_reload(new_mode_config, current_mode)
                return

            validate_changes = categorized[ReloadStrategy.VALIDATE_FIRST]
            if validate_changes:
                validation_results = self.hot_reload_manager.validate_changes(
                    validate_changes
                )
                if not all(validation_results.values()):
                    invalid = [f for f, v in validation_results.items() if not v]
                    logging.error(
                        "Validation failed for fields: %s. Aborting hot-reload.",
                        invalid,
                    )
                    return

            self.mode_config = new_mode_config
            self.mode_manager.config = new_mode_config

            self.current_config = self.mode_config.modes[
                current_mode
            ].to_runtime_config(self.mode_config)

            self.fuser = Fuser(self.current_config)
            logging.info("Fuser updated with new config.")

            if self.action_orchestrator:
                self.action_orchestrator.update_config(self.current_config)
                logging.info("ActionOrchestrator config updated.")

            if self.background_orchestrator:
                if hasattr(self.background_orchestrator, "update_config"):
                    self.background_orchestrator.update_config(self.current_config)
                else:
                    logging.warning(
                        "BackgroundOrchestrator does not support dynamic config update."
                    )

            if self.simulator_orchestrator:
                if hasattr(self.simulator_orchestrator, "update_config"):
                    self.simulator_orchestrator.update_config(self.current_config)
                else:
                    logging.warning(
                        "SimulatorOrchestrator does not support dynamic config update."
                    )

            if self.input_orchestrator:
                if hasattr(self.input_orchestrator, "update_config"):
                    self.input_orchestrator.update_config(self.current_config)
                else:
                    logging.warning(
                        "InputOrchestrator does not support dynamic config update."
                    )

            if hasattr(self.config_provider, "update_runtime_config"):
                self.config_provider.update_runtime_config(new_dict)
                logging.info("ConfigProvider persisted updated configuration.")
            else:
                logging.warning(
                    "ConfigProvider does not support persisting config updates."
                )

            for change in validate_changes + categorized[ReloadStrategy.HOT_RELOAD]:
                self.hot_reload_manager.track_change(change)
                logging.info(f"Hot-reloaded: {change.field_path} = {change.new_value}")

            logging.info("Selective hot-reload completed successfully.")

        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}", exc_info=True)
        finally:
            self._is_reloading = False

    async def _full_restart_reload(
        self, new_mode_config: ModeSystemConfig, current_mode: str
    ) -> None:
        await self._stop_current_orchestrators()

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
            f"config_reload->{current_mode}:full_restart"
        )

        await self._initialize_mode(current_mode)
        await self._start_orchestrators()

        logging.info(f"Full restart reload completed, active mode: {current_mode}")
