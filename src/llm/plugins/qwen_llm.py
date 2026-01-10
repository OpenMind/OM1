import json
import logging
import re
import time
import typing as T

import openai
from pydantic import BaseModel

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


class QwenLLM(LLM[R]):
    """
    Local Qwen LLM implementation using OpenAI-compatible API.

    Config example:
        "cortex_llm": {
            "type": "QwenLLM",
            "config": {
                "model": "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
            }
        }

    Parameters
    ----------
    config : LLMConfig
        Configuration with model name. Defaults to qwen30b-quantized.
    available_actions : list, optional
        List of available actions for function call generation.

    Attributes
    ----------
    _client : openai.AsyncClient
        Async client connected to local server at http://127.0.0.1:8000/v1.
    _extra_body : dict
        Extra parameters for Qwen (disables thinking mode).
    """

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        super().__init__(config, available_actions)

        if not config.model:
            self._config.model = "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"

        self._client = openai.AsyncClient(
            base_url="http://127.0.0.1:8000/v1",
            api_key="placeholder_key",
        )

        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.history_manager = LLMHistoryManager(self._config, self._client)

        self._skip_state_management = False

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, T.Any]] = []
    ) -> R | None:
        """
        Send prompt to local Qwen model and get structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send.
        messages : list of dict, optional
            Conversation history (default: []).

        Returns
        -------
        R or None
            Parsed response with actions, or None if parsing fails.
        """
        try:
            logging.info(f"Qwen input: {prompt}")
            logging.info(f"Qwen messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
            formatted.append({"role": "user", "content": prompt})

            model = self._config.model or "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"

            request_params: dict[str, T.Any] = {
                "model": model,
                "messages": formatted,
                "timeout": self._config.timeout,
                "extra_body": self._extra_body,
            }

            if self.function_schemas:
                request_params["tools"] = self.function_schemas
                request_params["tool_choice"] = "required"

            response = await self._client.chat.completions.create(**request_params)

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
                result = CortexOutputModel(actions=actions)
                return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"Qwen LLM error: {e}")
            return None
