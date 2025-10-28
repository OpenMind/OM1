import asyncio
import logging
from typing import List, Optional, Union

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.multi_mode.config import LifecycleHookType, ModeSystemConfig, RuntimeConfig
from runtime.multi_mode.manager import ModeManager
from simulators.orchestrator import SimulatorOrchestrator


class ModeCortexRuntime:
    """
    Mode-aware cortex runtime that can dynamically switch between different
    operational modes, each with their own configuration, inputs, and actions.
    """

    def __init__(self, mode_config: ModeSystemConfig):
        """
        Initialize the mode-aware cortex runtime.

        Parameters
        ----------
        mode_config : ModeSystemConfig
            The complete mode system configuration
        """
        self.mode_config = mode_config
        self.mode_manager = ModeManager(mode_config)
        self.io_provider = IOProvider()
        self.sleep_ticker_provider = SleepTickerProvider()

        # Current runtime components
        self.current_config: Optional[RuntimeConfig] = None
        self.fuser: Optional[Fuser] = None
        self.action_orchestrator: Optional[ActionOrchestrator] = None
        self.simulator_orchestrator: Optional[SimulatorOrchestrator] = None
        self.background_orchestrator: Optional[BackgroundOrchestrator] = None
        self.input_orchestrator: Optional[InputOrchestrator] = None

        # Tasks for orchestrators
        self.input_listener_task: Optional[asyncio.Task] = None
        self.simulator_task: Optional[asyncio.Future] = None
        self.action_task: Optional[asyncio.Future] = None
        self.background_task: Optional[asyncio.Future] = None

        # Setup transition callback
        self.mode_manager.add_transition_callback(self._on_mode_transition)

        # Flag to track if mode is initialized
        self._mode_initialized = False

    async def _initialize_mode(self, mode_name: str):
        """
        Initialize the runtime with a specific mode.

        Parameters
        ----------
        mode_name : str
            The name of the mode to initialize
        """
        mode_config = self.mode_config.modes[mode_name]

        mode_config.load_components(self.mode_config)

        self.current_config = mode_config.to_runtime_config(self.mode_config)

        logging.info(f"Initializing mode: {mode_config.display_name}")

        self.fuser = Fuser(self.current_config)
        self.action_orchestrator = ActionOrchestrator(self.current_config)
        self.simulator_orchestrator = SimulatorOrchestrator(self.current_config)
        self.background_orchestrator = BackgroundOrchestrator(self.current_config)

        logging.info(f"Mode '{mode_name}' initialized successfully")

    async def _on_mode_transition(self, from_mode: str, to_mode: str):
        """
        Handle mode transitions by gracefully stopping current components
        and starting new ones for the target mode.

        Parameters
        ----------
        from_mode : str
            The name of the mode being transitioned from
        to_mode : str
            The name of the mode being transitioned to
        """
        logging.info(f"Handling mode transition: {from_mode} -> {to_mode}")

        try:
            # Stop current orchestrators
            await self._stop_current_orchestrators()

            # Load new mode configuration
            await self._initialize_mode(to_mode)

            # Start new orchestrators
            await self._start_orchestrators()

            logging.info(f"Successfully transitioned to mode: {to_mode}")

        except Exception as e:
            logging.error(f"Error during mode transition {from_mode} -> {to_mode}: {e}")
            # TODO: Implement fallback/recovery mechanism
            raise

    async def _stop_current_orchestrators(self):
        """
        Stop all current orchestrator tasks gracefully.
        """
        logging.debug("Stopping current orchestrators...")

        tasks_to_cancel = []

        if self.input_listener_task and not self.input_listener_task.done():
            logging.debug("Cancelling input listener task")
            tasks_to_cancel.append(self.input_listener_task)

        if self.simulator_task and not self.simulator_task.done():
            logging.debug("Cancelling simulator task")
            tasks_to_cancel.append(self.simulator_task)

        if self.action_task and not self.action_task.done():
            logging.debug("Cancelling action task")
            tasks_to_cancel.append(self.action_task)

        if self.background_task and not self.background_task.done():
            logging.debug("Cancelling background task")
            tasks_to_cancel.append(self.background_task)

        # Cancel all tasks
        for task in tasks_to_cancel:
            task.cancel()

        # Wait for cancellations to complete
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                logging.debug(
                    f"Successfully cancelled {len(tasks_to_cancel)} orchestrator tasks"
                )
            except Exception as e:
                logging.warning(f"Error during orchestrator shutdown: {e}")

        # Clear task references
        self.input_listener_task = None
        self.simulator_task = None
        self.action_task = None
        self.background_task = None

        logging.debug("Orchestrators stopped successfully")

    async def _start_orchestrators(self):
        """
        Start orchestrators for the current mode.
        """
        if not self.current_config:
            raise RuntimeError("No current config available")

        # Start input listener
        self.input_orchestrator = InputOrchestrator(self.current_config.agent_inputs)
        self.input_listener_task = asyncio.create_task(self.input_orchestrator.listen())

        # Start other orchestrators
        if self.simulator_orchestrator:
            self.simulator_task = self.simulator_orchestrator.start()
        if self.action_orchestrator:
            self.action_task = self.action_orchestrator.start()
        if self.background_orchestrator:
            self.background_task = self.background_orchestrator.start()

        logging.debug("Orchestrators started successfully")

    async def _cleanup_tasks(self):
        """
        Cleanup all running tasks gracefully.
        """
        tasks_to_cancel = []

        if self.input_listener_task and not self.input_listener_task.done():
            tasks_to_cancel.append(self.input_listener_task)

        if self.simulator_task and not self.simulator_task.done():
            tasks_to_cancel.append(self.simulator_task)

        if self.action_task and not self.action_task.done():
            tasks_to_cancel.append(self.action_task)

        if self.background_task and not self.background_task.done():
            tasks_to_cancel.append(self.background_task)

        # Cancel all tasks
        for task in tasks_to_cancel:
            task.cancel()

        # Wait for cancellations to complete
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Error during final cleanup: {e}")

        logging.debug("Tasks cleaned up successfully")

    async def run(self) -> None:
        """
        Start the mode-aware runtime's main execution loop.
        """
        try:
            self.mode_manager.set_event_loop(asyncio.get_event_loop())

            if not self._mode_initialized:
                # Execute global startup hooks
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

                # Execute initial mode startup hooks
                initial_mode_config = self.mode_config.modes[
                    self.mode_manager.current_mode_name
                ]
                await initial_mode_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_STARTUP, startup_context
                )

            await self._start_orchestrators()

            cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

            while True:
                try:
                    awaitables: List[Union[asyncio.Task, asyncio.Future]] = [
                        cortex_loop_task
                    ]
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

                    if not cortex_loop_task.done():
                        continue
                    else:
                        break

                except Exception as e:
                    logging.error(f"Error in orchestrator tasks: {e}")
                    await asyncio.sleep(1.0)

        except Exception as e:
            logging.error(f"Error in mode-aware cortex runtime: {e}")
            raise
        finally:
            # Execute shutdown hooks before cleanup
            shutdown_context = {
                "system_name": self.mode_config.name,
                "final_mode": self.mode_manager.current_mode_name,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Execute current mode shutdown hooks
            current_config = self.mode_config.modes.get(
                self.mode_manager.current_mode_name
            )
            if current_config:
                await current_config.execute_lifecycle_hooks(
                    LifecycleHookType.ON_SHUTDOWN, shutdown_context
                )

            # Execute global shutdown hooks
            await self.mode_config.execute_global_lifecycle_hooks(
                LifecycleHookType.ON_SHUTDOWN, shutdown_context
            )

            await self._cleanup_tasks()

    async def _run_cortex_loop(self) -> None:
        """
        Execute the main cortex processing loop with mode awareness.
        """
        while True:
            try:
                if not self.sleep_ticker_provider.skip_sleep and self.current_config:
                    await self.sleep_ticker_provider.sleep(
                        1 / self.current_config.hertz
                    )

                await self._tick()
                self.sleep_ticker_provider.skip_sleep = False

            except Exception as e:
                logging.error(f"Error in cortex loop: {e}")
                await asyncio.sleep(1.0)

    async def _tick(self) -> None:
        """
        Execute a single tick of the mode-aware cortex processing cycle.
        """
        if not self.current_config or not self.fuser or not self.action_orchestrator:
            logging.warning("Cortex not properly initialized, skipping tick")
            return

        finished_promises, _ = await self.action_orchestrator.flush_promises()

        prompt = self.fuser.fuse(self.current_config.agent_inputs, finished_promises)
        if prompt is None:
            logging.debug("No prompt to fuse")
            return

        with self.io_provider.mode_transition_input():
            last_input = self.io_provider.get_mode_transition_input()
        new_mode = await self.mode_manager.process_tick(last_input)
        if new_mode:
            logging.info(f"Mode switched to: {new_mode}")
            return

        output = await self.current_config.cortex_llm.ask(prompt)
        if output is None:
            logging.debug("No output from LLM")
            return

        if self.simulator_orchestrator:
            await self.simulator_orchestrator.promise(output.actions)

        await self.action_orchestrator.promise(output.actions)

    def get_mode_info(self) -> dict:
        """
        Get information about the current mode and available transitions.
        """
        return self.mode_manager.get_mode_info()

    async def request_mode_change(self, target_mode: str) -> bool:
        """
        Request a manual mode change.

        Parameters
        ----------
        target_mode : str
            The name of the target mode

        Returns
        -------
        bool
            True if the transition was successful, False otherwise
        """
        return await self.mode_manager.request_transition(target_mode, "manual")

    def get_available_modes(self) -> dict:
        """
        Get information about all available modes.

        Returns
        -------
        dict
            Dictionary mapping mode names to their display information
        """
        return {
            name: {
                "display_name": config.display_name,
                "description": config.description,
                "is_current": name == self.mode_manager.current_mode_name,
            }
            for name, config in self.mode_config.modes.items()
        }
