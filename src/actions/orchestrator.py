import asyncio
import logging
import threading
import typing as T
from concurrent.futures import ThreadPoolExecutor

from actions.base import AgentAction
from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig


class ActionOrchestrator:
    """
    Manages data flow for the actions.

    Note: It is very important that the actions do not block the event loop.
    """

    promise_queue: T.List[asyncio.Task[T.Any]]
    _config: RuntimeConfig
    _connector_workers: int
    _connector_executor: ThreadPoolExecutor
    _submitted_connectors: T.Set[str]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self.promise_queue = []
        self._connector_workers = (
            min(12, len(config.agent_actions)) if config.agent_actions else 1
        )
        self._connector_executor = ThreadPoolExecutor(
            max_workers=self._connector_workers,
        )
        self._submitted_connectors = set()
        self._stop_event = threading.Event()

    def start(self):
        """
        Start actions and connectors in separate threads
        """
        for agent_action in self._config.agent_actions:
            if agent_action.llm_label in self._submitted_connectors:
                logging.warning(
                    f"Connector {agent_action.llm_label} already submitted, skipping."
                )
                continue
            self._connector_executor.submit(self._run_connector_loop, agent_action)
            self._submitted_connectors.add(agent_action.llm_label)

        return asyncio.Future()  # Return future for compatibility

    def _run_connector_loop(self, action: AgentAction):
        """
        Thread-based connector loop
        """
        while not self._stop_event.is_set():
            try:
                action.connector.tick()
            except Exception as e:
                logging.error(f"Error in connector {action.llm_label}: {e}")

    async def flush_promises(self) -> tuple[list[T.Any], list[asyncio.Task[T.Any]]]:
        """
        Flushes the promise queue and returns the completed promises and the pending promises.
        """
        done_promises = []
        for promise in self.promise_queue:
            if promise.done():
                await promise
                done_promises.append(promise)
        self.promise_queue = [p for p in self.promise_queue if p not in done_promises]
        return done_promises, self.promise_queue

    async def promise(self, actions: list[Action]) -> None:
        """
        Promises the actions to the appropriate connectors.

        Parameters
        ----------
        actions : list[Action]
            List of actions to promise to connectors.
        """
        for action in actions:
            logging.debug(f"Sending command: {action}")

            # fix corrupted commands when there is only one output
            # typically only happens during testing
            at = action.type.lower()
            av = action.value
            if at == "stand still" and av == "":
                action.type = "move"
                action.value = "stand still"
            elif at == "turn left" and av == "":
                action.type = "move"
                action.value = "turn left"
            elif at == "turn right" and av == "":
                action.type = "move"
                action.value = "turn right"
            elif at == "move forwards" and av == "":
                action.type = "move"
                action.value = "move forwards"
            elif at == "move back" and av == "":
                action.type = "move"
                action.value = "move back"

            agent_action = next(
                (
                    m
                    for m in self._config.agent_actions
                    if m.llm_label == action.type.lower()
                ),
                None,
            )
            if agent_action is None:
                logging.warning(
                    f"Attempted to call non-existent action: {action.type.lower()}."
                )
                continue
            action_response = asyncio.create_task(
                self._promise_action(agent_action, action)
            )
            self.promise_queue.append(action_response)

    async def _promise_action(self, agent_action: AgentAction, action: Action) -> T.Any:
        logging.debug(
            f"Calling action {agent_action.llm_label} with type {action.type.lower()} and argument {action.value}"
        )
        input_interface = T.get_type_hints(agent_action.interface)["input"](
            **{"action": action.value}
        )
        await agent_action.connector.connect(input_interface)
        return input_interface

    def stop(self):
        """
        Stop the action executor and wait for all tasks to complete.
        """
        self._stop_event.set()
        self._connector_executor.shutdown(wait=True)

    def __del__(self):
        """
        Clean up the ActionOrchestrator by stopping the executor.
        """
        self.stop()
