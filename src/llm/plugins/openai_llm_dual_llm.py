import asyncio
import json
import logging
import re
import time
import typing as T
from collections import deque

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _parse_qwen_tool_calls_from_text(text: str) -> list:
    """
    Container for LLM response with metadata.

    Wraps the result from an LLM call along with timing information
    and source identification for use in dual LLM comparison.

    Parameters
    ----------
    result : CortexOutputModel or None
        The parsed output model containing actions, or None if parsing failed.
    response_time : float
        Time in seconds taken to receive the response.
    source : str
        Identifier for the LLM source (e.g., "local" or "cloud").
    error : Exception, optional
        Any exception that occurred during the call, by default None.
    """
    tool_calls = []
    if not isinstance(text, str):
        return tool_calls

    matches = _QWEN_TOOL_CALL_RE.findall(text)
    for i, raw_json in enumerate(matches):
        try:
            obj = json.loads(raw_json)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if not name:
                continue
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        except Exception as e:
            logging.warning(f"Failed to parse Qwen tool_call JSON: {e}")
            continue

    return tool_calls


class LLMResponse:
    """Container for LLM response with metadata."""

    def __init__(
        self, result, response_time: float, source: str, error: Exception = None
    ):
        self.result = result
        self.response_time = response_time
        self.source = source
        self.error = error
        self.tool_calls = []
        self.content = None

        if result and hasattr(result, "actions"):
            self.tool_calls = result.actions


class OpenAILLM(LLM[R]):
    """
    Dual LLM implementation with timeout-based immediate selection.

    This class implements the LLM interface for OpenAI's GPT models, handling
    configuration, authentication, and async API communication. It supports both
    traditional JSON structured output and function calling.

    Races local and cloud LLMs within a timeout window:
    - If only one responds within timeout → use it immediately
    - If both respond within timeout → evaluate quality
    - If neither responds within timeout → fail fast, no waiting

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings. If not provided, defaults
        will be used.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided,
        the LLM will use function calls instead of structured JSON output.
    """

    def __init__(
        self,
        config: LLMConfig = LLMConfig(),
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the OpenAI LLM instance.

        Parameters
        ----------
        config : LLMConfig, optional
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "gpt-4.1-mini"

        # Auto-detect LLM mode based on configuration
        # Local LLM configuration
        self._local_base_url = getattr(config, "local_model_base_url", None)
        self._local_api_key = getattr(config, "local_model_api_key", None)
        self._local_model = getattr(config, "local_model", None)
        self._local_extra_body = getattr(config, "local_model_extra_body", None)

        # Cloud LLM configuration (uses main config)
        self._cloud_model = getattr(config, "cloud_model", config.model)
        self._timeout_threshold = getattr(config, "timeout_threshold", 3.2)

        # Determine which clients to create
        has_local = bool(
            self._local_base_url and self._local_api_key and self._local_model
        )
        has_cloud = bool(config.api_key)

        self._client = None
        self._local_client = None
        self._cloud_client = None

        if has_local and has_cloud:
            # Dual LLM mode (both configured)
            self._local_client = openai.AsyncClient(
                base_url=self._local_base_url,
                api_key=self._local_api_key,
            )
            self._cloud_client = openai.AsyncClient(
                base_url=config.base_url or "https://api.openmind.org/api/core/openai",
                api_key=config.api_key,
            )
            logging.debug(
                f"Dual LLM mode: local={self._local_model}, cloud={self._cloud_model}, "
                f"timeout={self._timeout_threshold}s"
            )
        elif has_local:
            # Local-only mode
            self._client = openai.AsyncClient(
                base_url=self._local_base_url,
                api_key=self._local_api_key,
            )
            self._config.model = self._local_model
            if self._local_extra_body:
                self._config.extra_body = self._local_extra_body
            logging.debug(f"Local LLM mode: {self._local_model}")
        elif has_cloud:
            # Cloud-only mode (default)
            self._client = openai.AsyncClient(
                base_url=config.base_url or "https://api.openmind.org/api/core/openai",
                api_key=config.api_key,
            )
            logging.debug(f"Cloud LLM mode: {self._cloud_model}")
        else:
            raise ValueError("No valid LLM configuration found")

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

        # Performance tracking
        self._response_times = deque(maxlen=30)
        self._total_requests = 0
        self._last_stats_log = time.time()
        self._stats_log_interval = 30.0

        # Dual LLM statistics
        self._local_wins = 0
        self._cloud_wins = 0
        self._both_timeout = 0
        self._evaluator_decisions = 0
        self._is_dual_mode = bool(self._local_client and self._cloud_client)

        logging.debug(f"Initialized OpenAILLM with model: {self._config.model}")
        logging.debug(f"Available actions: {len(available_actions or [])}")
        if hasattr(self, "function_schemas") and self.function_schemas:
            logging.debug(f"Generated {len(self.function_schemas)} function schemas")

    def _log_performance_stats(self, force: bool = False):
        """
        Log performance statistics periodically.

        Calculates and logs average, min, max, and percentile response times
        based on recent requests. Only logs if sufficient time has passed
        since the last log entry.

        Parameters
        ----------
        force : bool, optional
            If True, log stats regardless of time interval, by default False.
        """
        now = time.time()
        if not force and (now - self._last_stats_log) < self._stats_log_interval:
            return

        if not self._response_times:
            return

        times = list(self._response_times)
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        sorted_times = sorted(times)
        p50_idx = len(sorted_times) // 2
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)

        p50 = sorted_times[p50_idx]
        p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_time
        p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_time

        stats_msg = (
            f"LLM Performance Stats (last {len(times)} requests): "
            f"avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s, "
            f"p50={p50:.2f}s, p95={p95:.2f}s, p99={p99:.2f}s, "
            f"total_requests={self._total_requests}"
        )

        if self._is_dual_mode:
            stats_msg += (
                f" | Dual LLM: local_wins={self._local_wins}, "
                f"cloud_wins={self._cloud_wins}, "
                f"both_timeout={self._both_timeout}, "
                f"evaluator_decisions={self._evaluator_decisions}"
            )

        logging.debug(stats_msg)
        self._last_stats_log = now

    async def _call_single_llm(
        self, client: openai.AsyncClient, api_params: dict, source: str
    ) -> LLMResponse:
        """
        Call a single LLM and return wrapped response.

        Makes an async API call to the specified LLM client and wraps
        the response with timing and source metadata.

        Parameters
        ----------
        client : openai.AsyncClient
            The OpenAI async client to use for the API call.
        api_params : dict
            Parameters to pass to the chat completions API.
        source : str
            Identifier for the LLM source (e.g., "local" or "cloud").

        Returns
        -------
        LLMResponse
            Wrapped response containing result, timing, and metadata.
        """
        start_time = time.time()
        try:
            response = await client.chat.completions.create(**api_params)
            elapsed = time.time() - start_time

            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])
            logging.debug(f"{source} RAW response content: {message.content}")
            logging.debug(f"{source} RAW tool_calls: {message.tool_calls}")

            # Fallback parsing for Qwen
            if (
                not tool_calls
                and isinstance(message.content, str)
                and "<tool_call>" in message.content
            ):
                tool_calls = _parse_qwen_tool_calls_from_text(message.content)

            if tool_calls:
                function_call_data = [
                    {
                        "function": {
                            "name": (
                                tc.function.name
                                if hasattr(tc, "function")
                                else tc["function"]["name"]
                            ),
                            "arguments": (
                                tc.function.arguments
                                if hasattr(tc, "function")
                                else tc["function"]["arguments"]
                            ),
                        }
                    }
                    for tc in tool_calls
                ]
                actions = convert_function_calls_to_actions(function_call_data)
                result = CortexOutputModel(actions=actions)
            else:
                result = None

            llm_response = LLMResponse(result, elapsed, source)
            llm_response.content = message.content
            return llm_response

        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"{source} LLM error after {elapsed:.2f}s: {e}")
            return LLMResponse(None, elapsed, source, error=e)

    async def _evaluate_responses(
        self, local_resp: LLMResponse, cloud_resp: LLMResponse, original_prompt: str
    ) -> str:
        """
        Evaluate which response is better when both are within timeout.

        Uses LLM-based evaluation to compare response quality when both
        local and cloud models respond within the timeout window.

        Parameters
        ----------
        local_resp : LLMResponse
            Response from the local model.
        cloud_resp : LLMResponse
            Response from the cloud model.
        original_prompt : str
            The original user prompt/question for context.

        Returns
        -------
        str
            "local" or "cloud" indicating which response to use.
        """
        self._evaluator_decisions += 1

        # Quick check: if only one has tool calls, use that one
        local_has_tools = local_resp.result is not None
        cloud_has_tools = cloud_resp.result is not None

        if local_has_tools and not cloud_has_tools:
            logging.debug("Evaluator: Only local has tool calls → local")
            return "local"
        if cloud_has_tools and not local_has_tools:
            logging.debug("Evaluator: Only cloud has tool calls → cloud")
            return "cloud"

        if not local_has_tools and not cloud_has_tools:
            # Neither has tools, default to local
            logging.debug("Evaluator: Neither has tool calls → local (default)")
            return "local"

        # Both have tool calls - use LLM to evaluate quality
        try:
            # Format both responses for comparison
            local_actions = [
                {"type": a.type, "value": a.value} for a in local_resp.tool_calls
            ]
            cloud_actions = [
                {"type": a.type, "value": a.value} for a in cloud_resp.tool_calls
            ]

            # Include original prompt for better context
            evaluation_prompt = f"""You are evaluating two AI responses to determine which better answers the user's question.

Original User Question/Context:
{original_prompt[:500]}

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

            # Use local LLM as evaluator (faster and cheaper)
            eval_client = (
                self._local_client if self._local_client else self._cloud_client
            )
            eval_model = self._local_model if self._local_client else self._cloud_model

            logging.debug("LLM Evaluator: Comparing responses with original context...")

            response = await eval_client.chat.completions.create(
                model=eval_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.0,  # Deterministic
                max_tokens=10,
                timeout=2.0,  # Quick evaluation
            )

            result = response.choices[0].message.content.strip().upper()

            if "A" in result:
                logging.debug(
                    "LLM Evaluator chose: local (Response A) - better match to user question"
                )
                return "local"
            elif "B" in result:
                logging.debug(
                    "LLM Evaluator chose: cloud (Response B) - better match to user question"
                )
                return "cloud"
            else:
                logging.debug(
                    f"LLM Evaluator returned unexpected: '{result}' → defaulting to local"
                )
                return "local"

        except Exception as e:
            logging.error(f"LLM evaluation failed: {e} → defaulting to local")
            return "local"

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, T.Any]] = []
    ) -> R | None:
        """
        Send prompt to LLM(s) with immediate timeout-based selection.

        In dual LLM mode, races both local and cloud models and selects
        the best response based on timing and quality evaluation.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : list of dict, optional
            List of message dictionaries for conversation history,
            by default [].

        Returns
        -------
        R or None
            Parsed response matching the output_model structure,
            or None if parsing fails or no response is received.
        """
        request_start = time.time()

        try:
            logging.info(f"OpenAI LLM input: {prompt}")
            logging.debug(f"OpenAI LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            extra_body = getattr(self._config, "extra_body", None)
            extra_args: dict[str, T.Any] = {}
            if extra_body:
                extra_args["extra_body"] = extra_body

            # Build API parameters
            api_params = {
                "model": self._config.model or "gpt-5",
                "messages": T.cast(T.Any, formatted_messages),
                "timeout": self._config.timeout,
                **extra_args,
            }

            if self.function_schemas:
                api_params["tools"] = T.cast(T.Any, self.function_schemas)
                api_params["tool_choice"] = "auto"

            # Dual LLM with immediate timeout handling
            if self._local_client and self._cloud_client:
                logging.debug(
                    f"Racing local ({self._local_model}) and cloud ({self._cloud_model}) LLMs (timeout: {self._timeout_threshold}s)"
                )

                # Build local API params
                local_params = api_params.copy()
                local_params["model"] = self._local_model
                if self._local_extra_body:
                    local_params["extra_body"] = self._local_extra_body

                # Build cloud API params
                cloud_params = api_params.copy()
                cloud_params["model"] = self._cloud_model

                # Launch both tasks
                local_task = asyncio.create_task(
                    self._call_single_llm(self._local_client, local_params, "local")
                )
                cloud_task = asyncio.create_task(
                    self._call_single_llm(self._cloud_client, cloud_params, "cloud")
                )

                # Wait up to timeout_threshold for responses
                responses_in_time = []
                responses_timeout = []

                try:
                    # Race with timeout
                    done, pending = await asyncio.wait(
                        [local_task, cloud_task],
                        timeout=self._timeout_threshold,
                        return_when=asyncio.ALL_COMPLETED,  # Get all that finish in time
                    )

                    # Collect completed responses
                    for task in done:
                        try:
                            resp = task.result()
                            if resp.response_time <= self._timeout_threshold:
                                responses_in_time.append(resp)
                                logging.debug(
                                    f"{resp.source} responded in {resp.response_time:.2f}s ✓"
                                )
                            else:
                                responses_timeout.append(resp)
                                logging.debug(
                                    f"{resp.source} responded in {resp.response_time:.2f}s ✗ (exceeded timeout)"
                                )
                        except Exception as e:
                            logging.error(f"Task error: {e}")

                    # If responses are within timeout, cancel pending tasks
                    # If no responses in time, wait for pending to get fastest
                    if responses_in_time:
                        # We have at least one response in time, cancel slow ones
                        for task in pending:
                            task.cancel()
                            logging.warning(
                                f"Task cancelled - exceeded {self._timeout_threshold}s timeout"
                            )
                    else:
                        # No responses in time yet, wait for pending tasks to complete
                        if pending:
                            logging.debug(
                                "No responses within timeout, waiting for pending tasks..."
                            )
                            pending_done, _ = await asyncio.wait(
                                pending, return_when=asyncio.ALL_COMPLETED
                            )
                            for task in pending_done:
                                try:
                                    resp = task.result()
                                    responses_timeout.append(resp)
                                    logging.debug(
                                        f"{resp.source} responded in {resp.response_time:.2f}s ✗ (exceeded timeout)"
                                    )
                                except Exception as e:
                                    logging.error(f"Task error: {e}")

                except asyncio.TimeoutError:
                    # This shouldn't happen with asyncio.wait, but just in case
                    local_task.cancel()
                    cloud_task.cancel()
                    logging.warning("Both LLMs exceeded timeout")

                # Decision logic based on responses within timeout
                chosen_response = None

                if len(responses_in_time) == 2:
                    # Both responded in time - evaluate quality with LLM
                    local_resp = next(
                        r for r in responses_in_time if r.source == "local"
                    )
                    cloud_resp = next(
                        r for r in responses_in_time if r.source == "cloud"
                    )

                    winner = await self._evaluate_responses(
                        local_resp, cloud_resp, prompt
                    )
                    chosen_response = local_resp if winner == "local" else cloud_resp

                    if winner == "local":
                        self._local_wins += 1
                    else:
                        self._cloud_wins += 1

                    logging.debug(
                        f"Both in time: local={local_resp.response_time:.2f}s, "
                        f"cloud={cloud_resp.response_time:.2f}s → chose {winner}"
                    )

                elif len(responses_in_time) == 1:
                    # Only one responded in time - use it immediately
                    chosen_response = responses_in_time[0]

                    if chosen_response.source == "local":
                        self._local_wins += 1
                    else:
                        self._cloud_wins += 1

                    logging.debug(
                        f"Only {chosen_response.source} responded in time "
                        f"({chosen_response.response_time:.2f}s) → using {chosen_response.source}"
                    )

                else:
                    # Both exceeded timeout - use the faster one
                    if responses_timeout:
                        self._both_timeout += 1

                        # Find fastest response from timeout responses
                        chosen_response = min(
                            responses_timeout, key=lambda r: r.response_time
                        )

                        logging.debug(
                            f"Both LLMs exceeded {self._timeout_threshold}s timeout. "
                            f"Choosing faster: {chosen_response.source} ({chosen_response.response_time:.2f}s)"
                        )
                    else:
                        # Neither completed at all
                        self._both_timeout += 1
                        logging.warning(
                            f"Both LLMs exceeded {self._timeout_threshold}s timeout - no response available"
                        )
                        chosen_response = None

                if chosen_response:
                    final_result = chosen_response.result
                    total_time = chosen_response.response_time
                else:
                    final_result = None
                    total_time = self._timeout_threshold

            else:
                # Single LLM mode
                api_call_start = time.time()
                response = await self._client.chat.completions.create(**api_params)
                api_call_time = time.time() - api_call_start

                message = response.choices[0].message
                tool_calls = list(message.tool_calls or [])

                if (
                    not tool_calls
                    and isinstance(message.content, str)
                    and "<tool_call>" in message.content
                ):
                    tool_calls = _parse_qwen_tool_calls_from_text(message.content)

                if tool_calls:
                    function_call_data = [
                        {
                            "function": {
                                "name": (
                                    tc.function.name
                                    if hasattr(tc, "function")
                                    else tc["function"]["name"]
                                ),
                                "arguments": (
                                    tc.function.arguments
                                    if hasattr(tc, "function")
                                    else tc["function"]["arguments"]
                                ),
                            }
                        }
                        for tc in tool_calls
                    ]
                    actions = convert_function_calls_to_actions(function_call_data)
                    final_result = CortexOutputModel(actions=actions)
                else:
                    final_result = None

                total_time = time.time() - request_start

                logging.debug(
                    f"LLM Response Time: total={total_time:.2f}s, "
                    f"api_call={api_call_time:.2f}s, "
                    f"processing={total_time - api_call_time:.2f}s"
                )

            # Record stats
            self._response_times.append(total_time)
            self._total_requests += 1
            self._log_performance_stats()

            self.io_provider.llm_end_time = time.time()

            if final_result:
                logging.debug(f"OpenAI LLM function call output: {final_result}")
                return T.cast(R, final_result)

            logging.warning("No function calls returned by LLM")
            return None

        except Exception as e:
            error_time = time.time() - request_start
            self._response_times.append(error_time)
            self._total_requests += 1

            logging.error(
                f"OpenAI API error after {error_time:.2f}s: {e}", exc_info=True
            )
            return None

    def get_performance_summary(self) -> dict:
        """Get current performance statistics."""
        if not self._response_times:
            return {
                "total_requests": self._total_requests,
                "avg_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
            }

        times = list(self._response_times)
        summary = {
            "total_requests": self._total_requests,
            "avg_response_time": sum(times) / len(times),
            "min_response_time": min(times),
            "max_response_time": max(times),
            "sample_size": len(times),
        }

        if self._is_dual_mode:
            summary.update(
                {
                    "local_wins": self._local_wins,
                    "cloud_wins": self._cloud_wins,
                    "both_timeout": self._both_timeout,
                    "evaluator_decisions": self._evaluator_decisions,
                }
            )

        return summary
