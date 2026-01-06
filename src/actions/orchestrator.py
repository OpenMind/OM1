import asyncio
import logging
import threading
import time
import typing as T
from concurrent.futures import ThreadPoolExecutor

from actions.base import AgentAction
from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig


class ActionOrchestrator:
    """
    Manages data flow for the actions.

    Supports three execution modes:
    - concurrent: All actions execute simultaneously (default)
    - sequential: Actions execute one after another in order
    - dependencies: Actions wait for their dependencies to complete before executing

    Note: It is very important that the actions do not block the event loop.
    """

    promise_queue: T.List[asyncio.Task[T.Any]]
    _config: RuntimeConfig
    _connector_workers: int
    _connector_executor: ThreadPoolExecutor
    _submitted_connectors: T.Set[str]
    _stop_event: threading.Event
    _execution_mode: str
    _action_dependencies: T.Dict[str, T.List[str]]
    _completed_actions: T.Dict[str, asyncio.Event]

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
        self._execution_mode = config.action_execution_mode or "concurrent"
        self._action_dependencies = config.action_dependencies or {}
        self._completed_actions = {}

    def start(self):
        """
        Start actions and connectors in separate threads.
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
        Thread-based connector loop.
        """
        while not self._stop_event.is_set():
            try:
                action.connector.tick()
            except Exception as e:
                logging.error(f"Error in connector {action.llm_label}: {e}")
                time.sleep(0.1)

    async def flush_promises(self) -> tuple[list[T.Any], list[asyncio.Task[T.Any]]]:
        """
        Flushes the promise queue by waiting for all tasks to complete.
        Returns the completed promises and any remaining pending promises.

        Returns
        -------
        tuple[list[T.Any], list[asyncio.Task[T.Any]]]
            A tuple containing a list of completed promise results and a list of pending promise tasks.
        """
        if not self.promise_queue:
            return [], []

        done, pending = await asyncio.wait(
            self.promise_queue, return_when=asyncio.ALL_COMPLETED
        )

        self.promise_queue = []

        return list(done), list(pending)

    async def promise(self, actions: list[Action]) -> None:
        """
        Promises the actions to the appropriate connectors.

        Execution behavior depends on the configured execution mode:
        - concurrent: All actions execute simultaneously (default)
        - sequential: Actions execute one after another in the order provided
        - dependencies: Actions wait for their configured dependencies before executing

        Parameters
        ----------
        actions : list[Action]
            List of actions to promise to connectors.
        """
        self._completed_actions = {
            action.type.lower(): asyncio.Event() for action in actions
        }

        if self._execution_mode == "sequential":
            await self._promise_sequential(actions)
        elif self._execution_mode == "dependencies":
            await self._promise_with_dependencies(actions)
        else:
            await self._promise_concurrent(actions)

    async def _promise_concurrent(self, actions: list[Action]) -> None:
        """
        Execute all actions concurrently (original behavior).

        Parameters
        ----------
        actions : list[Action]
            List of actions to promise to connectors.
        """
        for action in actions:
            logging.debug(f"Sending command: {action}")
            action = self._normalize_action(action)

            agent_action = self._get_agent_action(action)
            if agent_action is None:
                continue

            action_response = asyncio.create_task(
                self._promise_action(agent_action, action)
            )
            self.promise_queue.append(action_response)

    async def _promise_sequential(self, actions: list[Action]) -> None:
        """
        Execute actions one after another in order.

        Parameters
        ----------
        actions : list[Action]
            List of actions to promise to connectors.
        """
        for action in actions:
            logging.debug(f"Sending command (sequential): {action}")
            action = self._normalize_action(action)

            agent_action = self._get_agent_action(action)
            if agent_action is None:
                continue

            action_response = asyncio.create_task(
                self._promise_action(agent_action, action)
            )
            self.promise_queue.append(action_response)
            await action_response

            action_label = action.type.lower()
            if action_label in self._completed_actions:
                self._completed_actions[action_label].set()

    async def _promise_with_dependencies(self, actions: list[Action]) -> None:
        """
        Execute actions respecting their dependencies.
        Actions wait for their prerequisites to complete before starting.

        Parameters
        ----------
        actions : list[Action]
            List of actions to promise to connectors.
        """
        for action in actions:
            logging.debug(f"Sending command (with dependencies): {action}")
            action = self._normalize_action(action)

            agent_action = self._get_agent_action(action)
            if agent_action is None:
                continue

            action_response = asyncio.create_task(
                self._promise_action_with_deps(agent_action, action)
            )
            self.promise_queue.append(action_response)

    async def _promise_action_with_deps(
        self, agent_action: AgentAction, action: Action
    ) -> T.Any:
        """
        Execute an action after waiting for its dependencies.

        Parameters
        ----------
        agent_action : AgentAction
            The agent action to execute.
        action : Action
            The action details.

        Returns
        -------
        T.Any
            The result of the action execution.
        """
        action_label = action.type.lower()
        dependencies = self._action_dependencies.get(action_label, [])

        for dep in dependencies:
            if dep in self._completed_actions:
                logging.debug(f"Action '{action_label}' waiting for dependency '{dep}'")
                await self._completed_actions[dep].wait()

        result = await self._promise_action(agent_action, action)

        if action_label in self._completed_actions:
            self._completed_actions[action_label].set()
            logging.debug(f"Action '{action_label}' completed")

        return result

    def _normalize_action(self, action: Action) -> Action:
        """
        Normalize action shortcuts to their full form.

        Parameters
        ----------
        action : Action
            The action to normalize.

        Returns
        -------
        Action
            The normalized action.
        """
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
        return action

    def _get_agent_action(self, action: Action) -> T.Optional[AgentAction]:
        """
        Get the agent action for a given action type.

        Parameters
        ----------
        action : Action
            The action to find the corresponding agent action for.

        Returns
        -------
        Optional[AgentAction]
            The corresponding AgentAction or None if not found.
        """
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
        return agent_action

    async def _promise_action(self, agent_action: AgentAction, action: Action) -> T.Any:
        """
        Promise a single action to its connector.

        Parameters
        ----------
        agent_action : AgentAction
            The agent action to execute.
        action : Action
            The action details.

        Returns
        -------
        T.Any
            The result of the action execution.
        """
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
