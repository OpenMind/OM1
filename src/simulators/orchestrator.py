import asyncio
import logging
import threading
import typing as T
from concurrent.futures import ThreadPoolExecutor

from llm.output_model import Action
from runtime.config import RuntimeConfig
from simulators.base import Simulator


class SimulatorOrchestrator:
    """
    Manages data flow to one or more simulators.

    Note: It is important that the simulators do not block the event loop.
    """

    promise_queue: T.List[asyncio.Task[T.Any]]
    _config: RuntimeConfig
    _simulator_workers: int
    _simulator_executor: ThreadPoolExecutor
    _simulator_instances: T.List[Simulator]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the Simulator Orchestrator.

        Parameters
        ----------
        config : RuntimeConfig
            Runtime configuration containing simulator settings.
        """
        self._config = config
        self.promise_queue = []
        self._simulator_workers = (
            min(12, len(config.simulators)) if config.simulators else 1
        )
        self._simulator_executor = ThreadPoolExecutor(
            max_workers=self._simulator_workers,
        )
        self._simulator_instances = []
        self._stop_event = threading.Event()

    def start(self):
        """
        Start simulators in separate threads.
        """
        for simulator in self._config.simulators:
            if any(sim.name == simulator.name for sim in self._simulator_instances):
                logging.warning(
                    f"Simulator {simulator.name} already submitted, skipping."
                )
                continue

            simulator.set_stop_event(self._stop_event)

            self._simulator_executor.submit(self._run_simulator_loop, simulator)
            self._simulator_instances.append(simulator)

        return asyncio.Future()

    def _run_simulator_loop(self, simulator: Simulator):
        """
        Thread-based simulator loop.

        Parameters
        ----------
        simulator : Simulator
            The simulator to run
        """
        while not self._stop_event.is_set():
            try:
                simulator.tick()
            except Exception:
                logging.exception(f"Error in simulator {simulator.name}")
                self._stop_event.wait(timeout=0.1)

    async def flush_promises(self) -> tuple[list[T.Any], list[asyncio.Task[T.Any]]]:
        """
        Flushes the promise queue and returns the completed promises and the pending promises.

        Returns
        -------
        tuple[list[Any], list[asyncio.Task[Any]]]
            A tuple containing the completed promises and the pending promises
        """
        done_promises = []
        for promise in self.promise_queue:
            if promise.done():
                await promise
                done_promises.append(promise)
        self.promise_queue = [p for p in self.promise_queue if p not in done_promises]
        return done_promises, self.promise_queue

    async def promise(self, actions: T.List[Action]) -> None:
        """
        Send actions to all simulators.

        Parameters
        ----------
        actions : list[Action]
            List of actions to send to the simulators
        """
        for simulator in self._config.simulators:
            simulator_response = asyncio.create_task(
                self._promise_simulator(simulator, actions)
            )
            self.promise_queue.append(simulator_response)

    async def _promise_simulator(
        self, simulator: Simulator, actions: T.List[Action]
    ) -> T.Any:
        """
        Send actions to a single simulator.

        Parameters
        ----------
        simulator : Simulator
            The simulator to send actions to
        actions : list[Action]
            List of actions to send to the simulator

        Returns
        -------
        Any
            The result of the simulator's response
        """
        logging.debug(f"Calling simulator {simulator.name} with actions {actions}")
        simulator.sim(actions)
        return None

    def stop(self):
        """
        Stop the simulator executor and wait for all tasks to complete.

        Sets the stop event to signal all simulator loops to terminate,
        calls stop() on each simulator instance for cleanup, then shuts
        down the thread pool executor and waits for all running tasks to
        finish gracefully.
        """
        self._stop_event.set()

        for simulator in self._simulator_instances:
            try:
                simulator.stop()
            except Exception:
                logging.exception(f"Error stopping simulator {simulator.name}")

        self._simulator_executor.shutdown(wait=True)
        self._simulator_instances.clear()

    def __del__(self):
        """
        Clean up the SimulatorOrchestrator by stopping the executor.
        """
        self.stop()
