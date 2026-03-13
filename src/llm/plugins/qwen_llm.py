import json
import logging
import re
import time
import typing as T

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)

_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_qwen_tool_calls(text: str) -> list:
    """
    Parse Qwen-style tool call blocks from text.

    Parameters
    ----------
    text : str
        Response text containing <tool_call>{...}</tool_call> blocks.

    Returns
    -------
    list
        List of parsed tool call dictionaries.
    """
    tool_calls = []
    if not isinstance(text, str):
        return tool_calls
    for i, raw in enumerate(_QWEN_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(raw)
            if name := obj.get("name"):
                tool_calls.append(
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                obj.get("arguments", {}), ensure_ascii=False
                            ),
                        },
                    }
                )
        except Exception:
            continue
    return tool_calls


class QwenLLMConfig(LLMConfig):
    """
    Configuration for Qwen LLM.

    Parameters
    ----------
    base_url : str
        Base URL for local Qwen API (default: http://127.0.0.1:8860/v1).
    api_key: str
        API key for local Qwen API (if required, default: "placeholder").
    model : str
        Qwen model name (e.g., "RedHatAI/Qwen3-30B-A3B-quantized.w4a16").
    enable_reasoning : bool
        Enable reasoning mode with more detailed thought processes in responses (default: False).
    """

    base_url: T.Optional[str] = Field(
        default="http://127.0.0.1:8860/v1", description="Base URL for local Qwen API"
    )
    api_key: T.Optional[str] = Field(
        default="placeholder", description="API key for local Qwen API (if required)"
    )
    model: T.Optional[str] = Field(
        default="RedHatAI/Qwen3-30B-A3B-quantized.w4a16", description="Qwen model name"
    )
    enable_reasoning: bool = Field(
        default=False,
        description="Enable reasoning mode with more detailed thought processes in responses",
    )


class QwenLLM(LLM[R]):
    """
    Local Qwen LLM implementation using OpenAI-compatible API.
    """

    def __init__(
        self,
        config: QwenLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the QwenLLM instance.

        Sets up the async client for the local Qwen model, configures extra body parameters,
        and initializes the history manager.

        Parameters
        ----------
        config : QwenLLMConfig
            Configuration settings for the LLM, including the model name.
        available_actions : list, optional
            List of available actions for function call generation.
        """
        super().__init__(config, available_actions)

        self._config: QwenLLMConfig = config

        self._base_url = self._config.base_url
        self._api_key = self._config.api_key
        self._model = self._config.model
        self._enable_reasoning = self._config.enable_reasoning

        self._client = openai.AsyncClient(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.history_manager = LLMHistoryManager(self._config, self._client)

        self._skip_state_management = False

    def _get_request_overrides(self) -> dict[str, T.Any]:
        """Return additional request params from config (temperature, max_tokens, etc.)."""
        overrides: dict[str, T.Any] = {}

        # LLMConfig supports both explicit `extra_params` and Pydantic "extra" fields.
        try:
            if isinstance(getattr(self._config, "extra_params", None), dict):
                overrides.update(self._config.extra_params)
        except Exception:
            pass

        try:
            pydantic_extra = getattr(self._config, "__pydantic_extra__", None)
            if isinstance(pydantic_extra, dict):
                overrides.update(pydantic_extra)
        except Exception:
            pass

        # Only forward a small, known-safe subset of OpenAI request parameters.
        # Runtime config injects meta keys like `mode`, `URID`, etc. which MUST NOT
        # be forwarded to the OpenAI client.
        allowlist = {
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
            "n",
            "stream",
            "response_format",
            "parallel_tool_calls",
            # We set tool_choice internally when tools are present; allow opt-in only
            # when no tools are configured.
            "tool_choice",
            "extra_body",
        }

        sanitized: dict[str, T.Any] = {
            key: value
            for key, value in overrides.items()
            if key in allowlist and value is not None
        }

        # Merge nested extra_body dicts instead of overwriting.
        if isinstance(sanitized.get("extra_body"), dict):
            sanitized_extra_body = sanitized.pop("extra_body")
            merged = dict(self._extra_body)
            merged.update(sanitized_extra_body)
            sanitized["extra_body"] = merged

        # Don't allow config to relax tool enforcement when tools are configured.
        if self.function_schemas and "tool_choice" in sanitized:
            sanitized.pop("tool_choice", None)

        return sanitized

    def _fallback_actions_from_text(self, text: str) -> list:
        """Best-effort fallback when backend fails to emit tool_calls.

        Only applies when exactly one tool is available (e.g. greeting mode with `speak`).
        """
        if not self.function_schemas or len(self.function_schemas) != 1:
            return []
        tool_name = (
            self.function_schemas[0].get("function", {}).get("name")
            if isinstance(self.function_schemas[0], dict)
            else None
        )
        if not tool_name:
            return []

        cleaned = (text or "").strip()
        if not cleaned:
            return []

        # Some backends may include tool-call tags in content; strip them.
        cleaned = _QWEN_TOOL_CALL_RE.sub("", cleaned).strip()
        if not cleaned:
            return []

        function_call_data = [
            {
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"action": cleaned}, ensure_ascii=False),
                }
            }
        ]
        return convert_function_calls_to_actions(function_call_data)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.Optional[T.List[T.Dict[str, T.Any]]] = None
    ) -> R | None:
        """
        Send prompt to local Qwen model and get structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send.
        messages : list of dict, optional
            Conversation history.

        Returns
        -------
        R or None
            Parsed response with actions, or None if parsing fails.
        """
        if messages is None:
            messages = []
        try:
            logging.info(f"Qwen input: {prompt}")
            logging.info(f"Qwen messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
            user_content = prompt if self._enable_reasoning else f"{prompt} /no_think"
            formatted.append({"role": "user", "content": user_content})

            request_params: dict[str, T.Any] = {
                "model": self._model,
                "messages": formatted,
                "timeout": self._config.timeout,
                "extra_body": self._extra_body,
            }

            request_params.update(self._get_request_overrides())

            if self.function_schemas:
                request_params["tools"] = self.function_schemas
                request_params["tool_choice"] = "required"

            response = await self._client.chat.completions.create(**request_params)

            if not response.choices:
                logging.warning("Qwen API returned empty choices")
                return None

            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()

            tool_calls = list(message.tool_calls or [])
            if (
                not tool_calls
                and isinstance(message.content, str)
                and "<tool_call>" in message.content
            ):
                tool_calls = _parse_qwen_tool_calls(message.content)

            if tool_calls:
                logging.info(f"Received {len(tool_calls)} function calls")
                logging.info(f"Function calls: {tool_calls}")

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
                if actions:
                    result = CortexOutputModel(actions=actions)
                    return T.cast(R, result)

                # If tool calls exist but arguments were malformed, fall back to text.
                if isinstance(message.content, str):
                    fallback_actions = self._fallback_actions_from_text(message.content)
                    if fallback_actions:
                        logging.warning(
                            "Qwen backend returned tool_calls but no valid actions; falling back to single-tool text execution"
                        )
                        result = CortexOutputModel(actions=fallback_actions)
                        return T.cast(R, result)

                return None

            # No tool calls: some OpenAI-compatible servers ignore tool_choice='required'.
            if isinstance(message.content, str):
                fallback_actions = self._fallback_actions_from_text(message.content)
                if fallback_actions:
                    logging.warning(
                        "Qwen backend returned no tool_calls; falling back to single-tool text execution"
                    )
                    result = CortexOutputModel(actions=fallback_actions)
                    return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"Qwen LLM error: {e}")
            return None
