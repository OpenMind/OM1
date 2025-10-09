"""
Mode-Aware Cortex Runtime for OM1

This module extends the basic CortexRuntime to support dynamic mode switching,
where different modes can have different inputs, actions, prompts, and behaviors.
"""

import asyncio
import logging
from typing import List, Optional, Union

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.config import RuntimeConfig
from runtime.mode_config import ModeSystemConfig
from runtime.mode_manager import ModeManager
from simulators.orchestrator import SimulatorOrchestrator


class ModeAwareCortexRuntime:
    """
    Mode-aware cortex runtime that can dynamically switch between different
    operational modes, each with their own configuration, inputs, and actions.

    This extends the basic CortexRuntime to support:
    - Dynamic mode switching
    - Mode-specific configurations
    - Graceful transition handling
    - Automatic mode transitions based on triggers
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

        # Current runtime components (will be updated on mode switches)
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

        # Initialize with the default mode
        asyncio.create_task(self._initialize_mode(self.mode_manager.current_mode_name))

    async def _initialize_mode(self, mode_name: str):
        """Initialize the runtime with a specific mode."""
        mode_config = self.mode_config.modes[mode_name]
        self.current_config = mode_config.to_runtime_config(self.mode_config)

        logging.info(f"Initializing mode: {mode_config.display_name}")

        # Create new components for this mode
        self.fuser = Fuser(self.current_config)
        self.action_orchestrator = ActionOrchestrator(self.current_config)
        self.simulator_orchestrator = SimulatorOrchestrator(self.current_config)
        self.background_orchestrator = BackgroundOrchestrator(self.current_config)

        logging.info(f"Mode '{mode_name}' initialized successfully")

    async def _on_mode_transition(self, from_mode: str, to_mode: str):
        """
        Handle mode transitions by gracefully stopping current components
        and starting new ones for the target mode.
        """
        logging.info(f"Handling mode transition: {from_mode} -> {to_mode}")

        try:
            # Stop current orchestrators gracefully
            await self._stop_current_orchestrators()

            # Initialize new mode
            await self._initialize_mode(to_mode)

            # Start new orchestrators
            await self._start_orchestrators()

            # Play transition messages if enabled
            if self.mode_config.transition_announcement:
                to_config = self.mode_config.modes[to_mode]
                if to_config.entry_message:
                    # TODO: Send entry message to speech action if available
                    logging.info(f"Mode entry: {to_config.entry_message}")

            logging.info(f"Successfully transitioned to mode: {to_mode}")

        except Exception as e:
            logging.error(f"Error during mode transition {from_mode} -> {to_mode}: {e}")
            # TODO: Implement fallback/recovery mechanism

    async def _stop_current_orchestrators(self):
        """Stop all current orchestrator tasks gracefully."""
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
                logging.warning(f"Error during orchestrator shutdown: {e}")

        logging.debug("Orchestrators stopped successfully")

    async def _start_orchestrators(self):
        """Start orchestrators for the current mode."""
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

    async def run(self) -> None:
        """
        Start the mode-aware runtime's main execution loop.
        """
        try:
            # Start initial orchestrators
            await self._start_orchestrators()

            # Run the main cortex loop
            cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

            # Gather all non-None tasks
            awaitables: List[Union[asyncio.Task, asyncio.Future]] = [cortex_loop_task]
            if self.input_listener_task:
                awaitables.append(self.input_listener_task)
            if self.simulator_task:
                awaitables.append(self.simulator_task)
            if self.action_task:
                awaitables.append(self.action_task)
            if self.background_task:
                awaitables.append(self.background_task)

            # Wait for all tasks
            await asyncio.gather(*awaitables)

        except Exception as e:
            logging.error(f"Error in mode-aware cortex runtime: {e}")
            raise

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
                # Continue running even if individual ticks fail
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def _tick(self) -> None:
        """
        Execute a single tick of the mode-aware cortex processing cycle.
        """
        if not self.current_config or not self.fuser or not self.action_orchestrator:
            logging.warning("Cortex not properly initialized, skipping tick")
            return

        # Collect all the latest inputs
        finished_promises, _ = await self.action_orchestrator.flush_promises()

        # Combine those inputs into a suitable prompt
        prompt = self.fuser.fuse(self.current_config.agent_inputs, finished_promises)
        if prompt is None:
            logging.debug("No prompt to fuse")
            return

        # Check for mode transitions based on the prompt/input
        last_input = getattr(self.io_provider, "_last_input_text", None)
        new_mode = await self.mode_manager.process_tick(last_input)
        if new_mode:
            logging.info(f"Mode switched to: {new_mode}")
            # The transition callback will handle the mode switch
            return

        # If there is a prompt, send to the AIs
        output = await self.current_config.cortex_llm.ask(prompt)
        if output is None:
            logging.debug("No output from LLM")
            return

        # Trigger the simulators
        if self.simulator_orchestrator:
            await self.simulator_orchestrator.promise(output.actions)

        # Trigger the actions
        await self.action_orchestrator.promise(output.actions)

    def get_mode_info(self) -> dict:
        """Get information about the current mode and available transitions."""
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
