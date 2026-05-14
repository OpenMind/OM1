# ParallelLLM: A system where N LLMs (any number: 1, 2, 3, or more) run in parallel,
# each generating specific actions they are capable of handling.
# Results are yielded as each LLM completes, allowing the cortex
# to execute actions immediately without waiting for all LLMs.

import asyncio
import logging
import time
import typing as T

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig, get_llm_class
from llm.output_model import Action, CortexOutputModel
from prometheus import om1_llm_latency, om1_llm_latency_last
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class LLMSpecConfig(BaseModel):
    """
    Configuration for a single LLM in the parallel system.

    Parameters
    ----------
    llm_type : str
        Class name of the LLM (e.g., "OpenAILLM", "QwenLLM").
    llm_config : LLMConfig
        Configuration for the LLM.
    action_filter : list[str], optional
        List of action names this LLM can generate.
        This filters the function call schemas sent to the LLM - the LLM will ONLY
        see and be able to call functions for actions in this list.
        If None or empty, the LLM can generate all available actions.
    """

    llm_type: str = Field(..., description="Class name of the LLM")
    config: LLMConfig = Field(default_factory=LLMConfig, description="Configuration for the LLM")
    action_filter: T.Optional[T.List[str]] = Field(
        default=None,
        description="List of action names this LLM can generate. "
        "Filters function call schemas - LLM only sees these functions. "
        "None means all actions.",
    )


class ParallelLLMConfig(LLMConfig):
    """
    Configuration for ParallelLLM.

    Parameters
    ----------
    llms : list[LLMSpecConfig]
        List of LLM specifications to run in parallel. Can be any number (1+).
    execute_immediately : bool
        If True, process results as each LLM completes (stream mode).
        If False, wait for all LLMs to complete before processing (batch mode).
        Default is True for streaming.
    """

    llms: T.List[LLMSpecConfig] = Field(..., description="List of LLM specifications to run in parallel")
    execute_immediately: bool = Field(
        default=True,
        description="Process results as each LLM completes (streaming mode)",
    )


class ParallelLLM(LLM[R]):
    """
    Parallel LLM that runs N LLMs concurrently with streaming result delivery.

    Each LLM can be configured to generate only specific action types,
    allowing specialized models to handle different aspects of the response.
    Each LLM maintains its own independent history manager for context tracking.

    Actions are yielded via ask_stream() as each LLM completes - no waiting.
    This allows true parallel execution where different LLMs handle different
    capabilities independently, and the orchestrator executes actions immediately
    as they become available.

    You can configure any number of LLMs (1, 2, 3, or more) based on your needs.

    Config example (3 LLMs):
        "cortex_llm": {
            "type": "ParallelLLM",
            "config": {
                "llms": [
                    {
                        "llm_type": "OpenAILLM",
                        "llm_config": {"model": "gpt-4.1"},
                        "action_filter": ["speak", "emotion"]
                    },
                    {
                        "llm_type": "QwenLLM",
                        "llm_config": {"model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"},
                        "action_filter": ["move", "navigate"]
                    },
                    {
                        "llm_type": "DeepSeekLLM",
                        "llm_config": {"model": "deepseek-chat"},
                        "action_filter": ["search", "analyze"]
                    }
                ],
                "execute_immediately": true
            }
        }
    """

    def __init__(
        self,
        config: ParallelLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the ParallelLLM instance.

        Sets up multiple LLMs based on configuration, filtering available
        actions for each LLM according to their action_filter specification.

        Parameters
        ----------
        config : ParallelLLMConfig
            Configuration settings for the Parallel LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: ParallelLLMConfig

        if not self._config.llms:
            raise ValueError("ParallelLLM requires at least one LLM configuration")

        self._llms: T.List[T.Tuple[LLM, T.Optional[T.List[str]]]] = []
        self._llm_label_to_name: T.Dict[str, str] = {}
        if available_actions:
            self._llm_label_to_name.update({a.llm_label: a.name for a in available_actions})

        # Initialize each LLM with filtered actions (supports N LLMs)
        for llm_spec in self._config.llms:
            llm_type = llm_spec.llm_type
            llm_cfg = llm_spec.config.model_copy()

            if not llm_cfg.api_key and self._config.api_key:
                llm_cfg.api_key = self._config.api_key
            if not llm_cfg.agent_name and self._config.agent_name:
                llm_cfg.agent_name = self._config.agent_name
            if not llm_cfg.history_length and self._config.history_length:
                llm_cfg.history_length = self._config.history_length
            if not llm_cfg.timeout and self._config.timeout:
                llm_cfg.timeout = self._config.timeout

            LLMClass = get_llm_class(llm_type)

            # Filter actions if action_filter is specified
            # This filters the function schemas sent to the LLM, so it only
            # sees and can call the specific function/actions in its filter.
            # The LLM will not have access to function call information for
            # actions outside its filter.
            filtered_actions = available_actions
            if llm_spec.action_filter and available_actions:
                filtered_actions = [action for action in available_actions if action.name in llm_spec.action_filter]
                logging.info(
                    f"Filtered function schemas for {llm_type}: "
                    f"{[a.name for a in filtered_actions]} "
                    f"(excluded {len(available_actions) - len(filtered_actions)} actions)"
                )
            elif available_actions:
                logging.info(f"No filter for {llm_type}, using all {len(available_actions)} actions")

            llm_instance: LLM = LLMClass(config=llm_cfg, available_actions=filtered_actions)

            self._llms.append((llm_instance, llm_spec.action_filter))

        client = None
        if self._llms:
            first_llm = self._llms[0][0]
            if hasattr(first_llm, "_client"):
                client = first_llm._client  # type: ignore

        if client is None:
            client = openai.AsyncClient(
                api_key=self._config.api_key or "dummy",
                base_url="http://localhost:8860/v1",
            )

        self.history_manager = LLMHistoryManager(self._config, client)

        logging.info(
            f"ParallelLLM initialized with {len(self._llms)} LLMs, "
            f"execute_immediately: {self._config.execute_immediately}."
        )

    async def _call_llm(
        self,
        llm: LLM,
        prompt: str,
        llm_name: str,
        action_filter: T.Optional[T.List[str]],
    ) -> dict:
        """
        Call an LLM and return result with timing info.

        Parameters
        ----------
        llm : LLM
            The LLM instance to call.
        prompt : str
            The prompt to send.
        llm_name : str
            Identifier for the LLM.
        action_filter : list[str], optional
            Filter for allowed action names. Used for safety validation only -
            the function schemas were already filtered when initializing the LLM.

        Returns
        -------
        dict
            Dictionary with keys: result, time, llm_name, action_filter.
        """
        start = time.time()
        try:
            result = await llm.ask(prompt)
            elapsed = time.time() - start

            # Safety check: Filter actions in case LLM somehow generated actions
            # outside its function schema (shouldn't happen, but validate anyway)
            if result and hasattr(result, "actions") and action_filter:
                filtered_actions = [
                    action for action in result.actions if self._llm_label_to_name.get(action.type) in action_filter
                ]
                if len(filtered_actions) != len(result.actions):
                    logging.warning(
                        f"{llm_name} generated {len(result.actions) - len(filtered_actions)} "
                        f"actions outside its function schema filter, removing them "
                        f"(this should not happen - function schemas were already filtered)"
                    )
                    result.actions = filtered_actions

            return {
                "result": result,
                "time": elapsed,
                "llm_name": llm_name,
                "action_filter": action_filter,
            }
        except Exception as e:
            logging.error(f"{llm_name} LLM error: {e}")
            return {
                "result": None,
                "time": time.time() - start,
                "llm_name": llm_name,
                "action_filter": action_filter,
            }

    def _merge_actions(self, results: T.List[dict]) -> T.List[Action]:
        """
        Collect actions from all LLM results (simple concatenation for final return).

        Parameters
        ----------
        results : list[dict]
            List of result dictionaries from _call_llm.

        Returns
        -------
        list[Action]
            Combined list of all actions.
        """
        all_actions = []

        for result in results:
            if result["result"] and hasattr(result["result"], "actions"):
                for action in result["result"].actions:
                    all_actions.append(action)

        logging.info(f"Collected {len(all_actions)} total actions from all LLMs")
        return all_actions

    async def ask_stream(
        self, prompt: str, messages: T.Optional[T.List[T.Dict[str, T.Any]]] = None
    ) -> T.AsyncGenerator[R, None]:
        """
        Send prompt to all configured LLMs in parallel and yield actions as each completes.

        This is an async generator that yields CortexOutputModel objects as soon as
        each LLM completes, allowing the orchestrator to execute actions immediately
        without waiting for all LLMs to finish. Works with any number of LLMs (1+).

        Parameters
        ----------
        prompt : str
            The input prompt to send to all LLMs.
        messages : list of dict, optional
            Conversation history

        Yields
        ------
        R
            CortexOutputModel containing actions from each LLM as it completes.
            Will yield N times for N LLMs configured (only if they have actions).
        """
        tasks = []
        try:
            llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            for idx, (llm, action_filter) in enumerate(self._llms):
                llm_name = f"{llm.__class__.__name__}_{idx}"
                task = asyncio.create_task(self._call_llm(llm, prompt, llm_name, action_filter))
                tasks.append(task)

            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result and not isinstance(result, Exception) and result["result"]:
                        actions = result["result"].actions
                        if actions:
                            logging.info(
                                f"{result['llm_name']} completed with "
                                f"{len(actions)} actions in {result['time']:.3f}s - "
                                f"yielding for immediate execution"
                            )
                            yield T.cast(R, result["result"])
                        else:
                            logging.info(f"{result['llm_name']} completed in " f"{result['time']:.3f}s with no actions")
                except Exception:
                    logging.exception("LLM task raised exception during streaming")

            latency = time.time() - llm_start_time
            om1_llm_latency.labels(model=str("parallel_llm"), endpoint=str("parallel_llm")).observe(latency)
            om1_llm_latency_last.labels(model=str("parallel_llm"), endpoint=str("parallel_llm")).set(latency)

        except asyncio.CancelledError:
            logging.info("ParallelLLM stream cancelled, cleaning up tasks")
            raise
        except Exception as e:
            logging.error(f"ParallelLLM stream error: {e}")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    @AvatarLLMState.trigger_thinking()
    async def ask(self, prompt: str, messages: T.Optional[T.List[T.Dict[str, T.Any]]] = None) -> R | None:
        """
        Send prompt to all configured LLMs in parallel and collect actions as they complete.

        If execute_immediately is True, returns actions as they arrive (uses streaming).
        Otherwise, waits for all LLMs and returns combined result. Works with any
        number of configured LLMs.

        Note: For immediate execution, consider using ask_stream() directly to get
        results as they complete rather than waiting for all LLMs.

        Parameters
        ----------
        prompt : str
            The input prompt to send to all LLMs.
        messages : list of dict, optional
            Conversation history.

        Returns
        -------
        R or None
            Combined response containing all actions from all configured LLMs, or None if all failed.
        """
        try:
            llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            tasks = []
            for idx, (llm, action_filter) in enumerate(self._llms):
                llm_name = f"{llm.__class__.__name__}_{idx}"
                task = asyncio.create_task(self._call_llm(llm, prompt, llm_name, action_filter))
                tasks.append(task)

            all_results = []

            # Process results as they arrive if execute_immediately is True (streaming mode)
            if self._config.execute_immediately:
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        if result and not isinstance(result, Exception):
                            all_results.append(result)

                            if result["result"] and hasattr(result["result"], "actions"):
                                actions = result["result"].actions
                                if actions:
                                    logging.info(
                                        f"{result['llm_name']} completed with "
                                        f"{len(actions)} actions in {result['time']:.3f}s - "
                                        f"actions ready for execution"
                                    )
                                else:
                                    logging.info(
                                        f"{result['llm_name']} completed in " f"{result['time']:.3f}s with no actions"
                                    )
                    except Exception as e:
                        logging.error(f"LLM task raised exception: {e}")

            else:
                # Batch mode: wait for all LLMs to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.error(f"LLM {idx} raised exception: {result}")
                    elif result is not None:
                        all_results.append(result)

                for result in all_results:
                    logging.info(f"{result['llm_name']} completed in {result['time']:.3f}s")

            if not all_results:
                logging.warning("All LLMs failed or returned no results")
                return None

            all_actions = self._merge_actions(all_results)

            if not all_actions:
                logging.warning("No actions generated by any LLM")
                return None

            latency = time.time() - llm_start_time
            om1_llm_latency.labels(model=str("parallel_llm"), endpoint=str("parallel_llm")).observe(latency)
            om1_llm_latency_last.labels(model=str("parallel_llm"), endpoint=str("parallel_llm")).set(latency)

            result = CortexOutputModel(actions=all_actions)
            return T.cast(R, result)

        except Exception as e:
            logging.error(f"ParallelLLM error: {e}")
            return None
