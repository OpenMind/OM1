import asyncio
import json
import logging
import re
import time
import typing as T

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig, get_llm_class
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


def _extract_voice_input(prompt: str) -> str:
    """
    Extract voice input from the prompt.

    Parameters
    ----------
    prompt : str
        Full prompt containing INPUT: Voice section.

    Returns
    -------
    str
        Extracted voice input text, or empty string if not found.
    """
    match = re.search(r"INPUT: Voice\s*// START\s*(.*?)\s*// END", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class DualLLMConfig(LLMConfig):
    """
    Configuration for DualLLM.

    Parameters
    ----------
    local_llm_type : str
        Class name of the local LLM (default: "QwenLLM").
    local_llm_config : dict
        Configuration for the local LLM.
    cloud_llm_type : str
        Class name of the cloud LLM (default: "OpenAILLM").
    cloud_llm_config : dict
        Configuration for the cloud LLM.
    """

    local_llm_type: str = Field(
        default="QwenLLM", description="Class name of the local LLM"
    )
    local_llm_config: T.Dict[str, T.Any] = Field(
        default_factory=lambda: {"model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"},
        description="Configuration for the local LLM",
    )
    cloud_llm_type: str = Field(
        default="OpenAILLM", description="Class name of the cloud LLM"
    )
    cloud_llm_config: T.Dict[str, T.Any] = Field(
        default_factory=lambda: {"model": "gpt-4.1"},
        description="Configuration for the cloud LLM",
    )


class DualLLM(LLM[R]):
    """
    Dual LLM that races local and cloud LLMs with three selection rules:
    1. Both in time → pick one with function calls, or evaluate quality if both have
    2. One in time → use it
    3. Neither in time → use first to complete.

    Config example:
        "cortex_llm": {
            "type": "DualLLM",
            "config": {
                "local_llm_type": "QwenLLM",
                "local_llm_config": {"model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"},
                "cloud_llm_type": "OpenAILLM",
                "cloud_llm_config": {"model": "gpt-4.1"}
            }
        }

    Parameters
    ----------
    config : LLMConfig, optional
        Configuration settings for the LLM.
    available_actions : list[AgentAction], optional
        List of available actions for function calling.
    """

    TIMEOUT_THRESHOLD = 3.2

    def __init__(
        self,
        config: DualLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the DualLLM instance.

        Sets up the local and cloud LLMs based on configuration, initializes
        the evaluation client, and prepares the history manager.

        Parameters
        ----------
        config : DualLLMConfig
            Configuration settings for the Dual LLM, including local and cloud LLM details.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: DualLLMConfig

        local_type = self._config.local_llm_type
        local_cfg = self._config.local_llm_config.copy()
        cloud_type = self._config.cloud_llm_type
        cloud_cfg = self._config.cloud_llm_config.copy()

        cloud_cfg["api_key"] = self._config.api_key

        LocalLLMClass = get_llm_class(local_type)
        CloudLLMClass = get_llm_class(cloud_type)

        self._local_llm: LLM = LocalLLMClass(
            config=LLMConfig(**local_cfg), available_actions=available_actions
        )
        self._cloud_llm: LLM = CloudLLMClass(
            config=LLMConfig(**cloud_cfg), available_actions=available_actions
        )

        self._local_llm._skip_state_management = True
        self._cloud_llm._skip_state_management = True

        self._eval_client = openai.AsyncClient(
            base_url="http://127.0.0.1:8000/v1", api_key="local"
        )
        self._eval_model = local_cfg.get(
            "model", "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
        )

        self.history_manager = LLMHistoryManager(self._config, self._eval_client)

    async def _call_llm(
        self, llm: LLM, prompt: str, messages: T.List[T.Dict[str, T.Any]], source: str
    ) -> dict:
        """
        Call an LLM and return result with timing info.

        Parameters
        ----------
        llm : LLM
            The LLM instance to call.
        prompt : str
            The prompt to send.
        source : str
            Identifier for the source ("local" or "cloud").

        Returns
        -------
        dict
            Dictionary with keys: result, time, source.
        """
        start = time.time()
        try:
            result = await llm.ask(prompt, messages)
            return {"result": result, "time": time.time() - start, "source": source}
        except Exception as e:
            logging.error(f"{source} LLM error: {e}")
            return {"result": None, "time": time.time() - start, "source": source}

    def _has_function_calls(self, entry: dict) -> bool:
        """
        Check if result has valid function calls.

        Parameters
        ----------
        entry : dict
            Result dictionary from _call_llm.

        Returns
        -------
        bool
            True if result contains valid actions.
        """
        result = entry["result"]
        return (
            result is not None and hasattr(result, "actions") and bool(result.actions)
        )

    async def _evaluate_quality(
        self, local_entry: dict, cloud_entry: dict, prompt: str
    ) -> str:
        """
        Use LLM to evaluate which response better answers the user's question.

        Parameters
        ----------
        local_entry : dict
            Result from local LLM.
        cloud_entry : dict
            Result from cloud LLM.
        voice_input : str
            Extracted user voice input for context.

        Returns
        -------
        str
            "local" or "cloud" indicating the better response.
        """
        try:
            local_actions = [
                {"type": a.type, "value": a.value}
                for a in local_entry["result"].actions
            ]
            cloud_actions = [
                {"type": a.type, "value": a.value}
                for a in cloud_entry["result"].actions
            ]

            eval_prompt = f"""You are evaluating two AI responses to determine which better answers the user's question.

Original User Question/Context:
{prompt[:500]}

Response A (local model):
{json.dumps(local_actions, indent=2)}

Response B (cloud model):
{json.dumps(cloud_actions, indent=2)}

Evaluate based on:
1. Relevance - Which response better addresses the user's question?
2. Completeness - Does it fully answer what was asked?
3. Appropriateness - Are the actions suitable for the context?
4. Quality - Is the content natural and engaging?

Respond with ONLY a single word: either "A" or "B" for the better response."""

            response = await self._eval_client.chat.completions.create(
                model=self._eval_model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0,
                max_tokens=10,
                timeout=2.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            if not response.choices:
                logging.warning("LLM evaluation returned empty choices")
                return "local"

            content = response.choices[0].message.content
            if content is None:
                return "local"

            result = content.strip().upper()
            return "local" if "A" in result else "cloud"
        except Exception as e:
            logging.warning(f"LLM quality evaluation failed, defaulting to local: {e}")
            return "local"

    async def _select_best(
        self, local_entry: dict, cloud_entry: dict, prompt: str
    ) -> dict:
        """
        Select best response when both LLMs respond in time.

        Parameters
        ----------
        local_entry : dict
            Result from local LLM.
        cloud_entry : dict
            Result from cloud LLM.
        prompt : str
            The prompt text for evaluation context.

        Returns
        -------
        dict
            The selected result entry.
        """
        local_has_function_call = self._has_function_calls(local_entry)
        cloud_has_function_call = self._has_function_calls(cloud_entry)

        if local_has_function_call and not cloud_has_function_call:
            return local_entry
        if cloud_has_function_call and not local_has_function_call:
            return cloud_entry
        if not local_has_function_call and not cloud_has_function_call:
            return local_entry

        # Both have function calls → evaluate quality
        winner = await self._evaluate_quality(local_entry, cloud_entry, prompt)
        return local_entry if winner == "local" else cloud_entry

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, T.Any]] = []
    ) -> R | None:
        """
        Send prompt to both LLMs and select the best response.

        Parameters
        ----------
        prompt : str
            The input prompt to send.
        messages : list of dict, optional
            Conversation history (default: []).

        Returns
        -------
        R or None
            Parsed response matching the output model, or None if failed.
        """
        try:
            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            voice_input = _extract_voice_input(prompt)

            local_task = asyncio.create_task(
                self._call_llm(self._local_llm, prompt, messages, "local")
            )
            cloud_task = asyncio.create_task(
                self._call_llm(self._cloud_llm, prompt, messages, "cloud")
            )
            tasks = {"local": local_task, "cloud": cloud_task}

            start_time = time.time()
            in_time = {}

            # Wait for responses until timeout
            while (
                len(in_time) < 2 and (time.time() - start_time) < self.TIMEOUT_THRESHOLD
            ):
                pending = [t for name, t in tasks.items() if name not in in_time]
                if not pending:
                    break

                remaining = self.TIMEOUT_THRESHOLD - (time.time() - start_time)
                if remaining <= 0:
                    break

                done, _ = await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    result = task.result()
                    if (
                        result["time"] <= self.TIMEOUT_THRESHOLD
                        and result["result"] is not None
                    ):
                        in_time[result["source"]] = result

            # Both in time → select best
            if len(in_time) == 2:
                logging.debug("Both LLMs responded in time, evaluating best response.")
                chosen = await self._select_best(
                    in_time["local"], in_time["cloud"], voice_input
                )

            # One in time → use it
            elif len(in_time) == 1:
                chosen = list(in_time.values())[0]
                logging.debug(
                    f"One LLM responded in time, using its response. {chosen['source']} LLM selected."
                )
                # Cancel the other task
                for name, task in tasks.items():
                    if name not in in_time:
                        task.cancel()
                        logging.debug(f"Cancelled {name} LLM task due to timeout.")

            # Neither in time → wait for first to complete
            else:
                logging.debug(
                    "Neither LLM responded in time, waiting for first to complete."
                )
                pending = [t for t in tasks.values() if not t.done()]
                if pending:
                    done, rest = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    chosen = list(done)[0].result()
                    logging.debug(
                        f"Using first completed LLM response from {chosen['source']} LLM."
                    )
                    for task in rest:
                        task.cancel()
                        logging.debug(f"Cancelled {task} LLM task due to timeout.")
                else:
                    # Both already completed (just late)
                    results = [t.result() for t in tasks.values()]
                    chosen = min(results, key=lambda x: x["time"])

            self.io_provider.llm_end_time = time.time()

            if chosen and chosen["result"]:
                return T.cast(R, chosen["result"])
            return None

        except Exception as e:
            logging.error(f"DualLLM error: {e}")
            return None
