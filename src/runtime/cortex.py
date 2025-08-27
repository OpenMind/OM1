import asyncio
import logging

from actions.orchestrator import ActionOrchestrator
from backgrounds.orchestrator import BackgroundOrchestrator
from fuser import Fuser
from inputs.orchestrator import InputOrchestrator
from providers.io_provider import IOProvider
from providers.sleep_ticker_provider import SleepTickerProvider
from runtime.config import RuntimeConfig
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

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the CortexRuntime with provided configuration.

        Parameters
        ----------
        config : RuntimeConfig
            Configuration object for the runtime.
        """
        self.config = config

        logging.debug(f"Cortex runtime config: {config}")
        self.fuser = Fuser(config)
        self.action_orchestrator = ActionOrchestrator(config)
        self.simulator_orchestrator = SimulatorOrchestrator(config)
        self.background_orchestrator = BackgroundOrchestrator(config)
        self.sleep_ticker_provider = SleepTickerProvider()
        self.io_provider = IOProvider()

        self.silence_counter = 0
        self.silence_rate = 0
        if hasattr(self.config, "silence_rate"):
            self.silence_rate = self.config.silence_rate

    async def run(self) -> None:
        """
        Start the runtime's main execution loop.

        This method initializes input listeners and begins the cortex
        processing loop, running them concurrently.

        Returns
        -------
        None
        """
        input_listener_task = await self._start_input_listeners()
        cortex_loop_task = asyncio.create_task(self._run_cortex_loop())

        simulator_start = self._start_simulator_task()
        action_start = self._start_action_task()
        background_start = self._start_background_task()

        await asyncio.gather(
            input_listener_task,
            cortex_loop_task,
            simulator_start,
            action_start,
            background_start,
        )

    async def _start_input_listeners(self) -> asyncio.Task:
        """
        Initialize and start input listeners.

        Creates an InputOrchestrator for the configured agent inputs
        and starts listening for input events.

        Returns
        -------
        asyncio.Task
            Task handling input listening operations.
        """
        input_orchestrator = InputOrchestrator(self.config.agent_inputs)
        input_listener_task = asyncio.create_task(input_orchestrator.listen())
        return input_listener_task

    async def _start_simulator_task(self) -> asyncio.Future:
        return self.simulator_orchestrator.start()

    async def _start_action_task(self) -> asyncio.Future:
        return self.action_orchestrator.start()

    async def _start_background_task(self) -> asyncio.Future:
        return self.background_orchestrator.start()

    async def _run_cortex_loop(self) -> None:
        """
        Execute the main cortex processing loop.

        Runs continuously, managing the sleep/wake cycle and triggering
        tick operations at the configured frequency.

        Returns
        -------
        None
        """
        while True:
            if not self.sleep_ticker_provider.skip_sleep:
                await self.sleep_ticker_provider.sleep(1 / self.config.hertz)
            await self._tick()
            self.sleep_ticker_provider.skip_sleep = False

    async def _tick(self) -> None:
        """
        Execute a single tick of the cortex processing cycle.

        Collects inputs, generates prompts, processes them through the LLM,
        and triggers appropriate simulators and actions based on the output.

        Returns
        -------
        None
        """
        # collect all the latest inputs
        finished_promises, _ = await self.action_orchestrator.flush_promises()

        # combine those inputs into a suitable prompt
        prompt = self.fuser.fuse(self.config.agent_inputs, finished_promises)
        if prompt is None:
            logging.warning("No prompt to fuse")
            return

        # if there is a prompt, send to the AIs
        output = await self.config.cortex_llm.ask(prompt)
        if output is None:
            logging.warning("No output from LLM")
            return

        # Trigger the simulators
        await self.simulator_orchestrator.promise(output.actions)

        actions_silent = []
        for action in output.actions:
            action_type = action.type.lower()
            if action_type != "speak" and action_type != "arm movement":
                actions_silent.append(action)
                logging.debug(f"appended: {action_type}")

        # Trigger actions
        if "INPUT: Voice" in prompt:
            logging.info("responding due to prior voice input")
            self.silence_counter = 0
            await self.action_orchestrator.promise(output.actions)
        elif self.silence_counter >= self.silence_rate:
            # speak at desired duty rate
            self.silence_counter = 0
            await self.action_orchestrator.promise(output.actions)
        else:
            # do not speak
            self.silence_counter += 1
            await self.action_orchestrator.promise(actions_silent)
