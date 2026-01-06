import logging
import os
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


def _flatten_content(x: T.Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return _flatten_content(x.get("content", ""))
    if isinstance(x, list):
        parts: T.List[str] = []
        for item in x:
            parts.append(_flatten_content(item))
        return "\n".join([p for p in parts if p])
    return str(x)


def _normalize_messages(raw: T.Any) -> T.List[T.Dict[str, str]]:
    """
    Accepts:
      - str prompt
      - dict message
      - list[dict] messages
      - list mixed
    Returns OpenAI-valid messages: [{'role': str, 'content': str}, ...]
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]

    if isinstance(raw, dict):
        return [{
            "role": raw.get("role", "user"),
            "content": _flatten_content(raw.get("content", "")),
        }]

    if isinstance(raw, list):
        out: T.List[T.Dict[str, str]] = []
        for msg in raw:
            if isinstance(msg, str):
                out.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                out.append({
                    "role": msg.get("role", "user"),
                    "content": _flatten_content(msg.get("content", "")),
                })
            else:
                out.append({"role": "user", "content": _flatten_content(msg)})
        return out

    return [{"role": "user", "content": _flatten_content(raw)}]


class OpenAILLM(LLM[R]):
    """
    IMPORTANT: Class name must match config: "type": "OpenAILLM"
    """

    def __init__(self, config: LLMConfig, available_actions: T.Optional[T.List] = None):
        super().__init__(config, available_actions)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        if not config.model:
            self._config.model = "gpt-4o-mini"

        # OpenAI direct
        self._client = openai.AsyncClient(api_key=api_key)

        # Decorators expect this attribute
        self.history_manager = LLMHistoryManager(self._config, self._client)

    @AvatarLLMState.trigger_thinking()
    async def ask(self, prompt: T.Any, messages: T.Optional[T.List[T.Dict[str, T.Any]]] = None) -> T.Optional[R]:
        """
        OM1 compatibility:
          - ask(prompt: str)
          - ask(messages: list[dict])  -> prompt is actually messages
          - ask(prompt: str, messages=[...]) -> concatenates
        """
        try:
            base_msgs = _normalize_messages(messages) if messages else []
            prompt_msgs = _normalize_messages(prompt)

            final_messages = base_msgs + prompt_msgs
            if not final_messages:
                logging.warning("Skipping LLM call: no messages")
                return None

            logging.info(f"OpenAI messages: {final_messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(final_messages[-1]["content"])

            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=T.cast(T.Any, final_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            self.io_provider.llm_end_time = time.time()

            message = response.choices[0].message

            if message.tool_calls:
                function_call_data = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
                actions = convert_function_calls_to_actions(function_call_data)
                return T.cast(R, CortexOutputModel(actions=actions))

            return None

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return None
