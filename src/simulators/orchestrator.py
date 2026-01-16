import asyncio
import logging
import threading
import typing as T
from concurrent.futures import ThreadPoolExecutor

from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig
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
    _submitted_simulators: T.Set[str]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self.promise_queue = []
        self._simulator_workers = (
            min(12, len(config.simulators)) if config.simulators else 1
        )
        self._simulator_executor = ThreadPoolExecutor(
            max_workers=self._simulator_workers,
        )
        self._submitted_simulators = set()
        self._stop_event = threading.Event()

    def start(self):
        """
        Start simulators in separate threads.
        """
        for simulator in self._config.simulators:
            if simulator.name in self._submitted_simulators:
                logging.warning(
                    f"Simulator {simulator.name} already submitted, skipping."
                )
                continue
            self._simulator_executor.submit(self._run_simulator_loop, simulator)
            self._submitted_simulators.add(simulator.name)

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
            except Exception as e:
                logging.error(f"Error in simulator {simulator.name}: {e}")

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
        """
        self._stop_event.set()
        self._simulator_executor.shutdown(wait=True)

    def __del__(self):
        """
        Clean up the SimulatorOrchestrator by stopping the executor.
        """
        self.stop()
