"""
Simplified Qwen LLM plugin for local models with limited tool-call support.

Builds Action objects directly (bypassing convert_function_calls_to_actions)
to preserve field names for single-arg interfaces. Includes a text fallback
for when the model returns plain text instead of tool calls.
"""

import json
import logging
import re
import time
import typing as T

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.output_model import Action, CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager_simplified import LLMHistoryManagerSimplified

R = T.TypeVar("R", bound=BaseModel)

_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_qwen_tool_calls(text: str) -> list:
    """Parse Qwen-style <tool_call>{...}</tool_call> blocks from text."""
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


class QwenLLMSimplifiedConfig(LLMConfig):
    """
    Configuration for simplified Qwen LLM.

    Parameters
    ----------
    base_url : str
        Base URL for local Qwen API (default: http://127.0.0.1:8860/v1).
    api_key: str
        API key for local Qwen API (if required, default: "placeholder").
    model : str
        Qwen model name.
    enable_reasoning : bool
        Enable reasoning mode (default: False).
    """

    base_url: T.Optional[str] = Field(
        default="http://127.0.0.1:8860/v1",
        description="Base URL for local Qwen API",
    )
    api_key: T.Optional[str] = Field(
        default="placeholder",
        description="API key for local Qwen API (if required)",
    )
    model: T.Optional[str] = Field(
        default="RedHatAI/Qwen3-30B-A3B-quantized.w4a16",
        description="Qwen model name",
    )
    enable_reasoning: bool = Field(
        default=False,
        description="Enable reasoning mode",
    )


class QwenLLMSimplified(LLM[R]):
    """
    Simplified Qwen LLM that builds Actions directly.

    Bypasses convert_function_calls_to_actions to preserve field names
    for single-arg interfaces. Includes a text fallback that wraps
    plain text as a {"response": text} action.
    """

    def __init__(
        self,
        config: QwenLLMSimplifiedConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        super().__init__(config, available_actions)

        self._config: QwenLLMSimplifiedConfig = config
        self._base_url = self._config.base_url
        self._api_key = self._config.api_key
        self._model = self._config.model
        self._enable_reasoning = self._config.enable_reasoning

        self._client = openai.AsyncClient(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.history_manager = LLMHistoryManagerSimplified(self._config, self._client)

        self._skip_state_management = False

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManagerSimplified.update_history()
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

                # Build Actions directly, preserving argument field names
                # via json.dumps (bypasses convert_function_calls_to_actions
                # which loses field names for single-arg calls).
                actions = []
                for tc in tool_calls:
                    fn_name = (
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc["function"]["name"]
                    )
                    fn_args = (
                        tc.function.arguments
                        if hasattr(tc, "function")
                        else tc["function"]["arguments"]
                    )
                    # Ensure value is a JSON string with field names preserved
                    if isinstance(fn_args, str):
                        try:
                            args = json.loads(fn_args)
                            fn_args = json.dumps(args, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass
                    else:
                        fn_args = json.dumps(fn_args, ensure_ascii=False)

                    # Skip tool calls whose response is a conversation summary
                    val_lower = fn_args.lower()
                    if "conversation summary" in val_lower or (
                        "previously," in val_lower and "user" in val_lower
                    ):
                        logging.info(f"Skipping summary tool call: {fn_args[:200]}")
                        continue

                    actions.append(Action(type=fn_name, value=fn_args))
                if actions:
                    result = CortexOutputModel(actions=actions)
                    return T.cast(R, result)
                logging.info("All tool calls were summary-style, returning None")
                return None

            # Fallback: if model returned text but no tool calls, and there's
            # exactly one function schema, wrap the text as a function call
            text = message.content
            if text and self.function_schemas and len(self.function_schemas) == 1:
                # Skip summary-style responses that aren't user-facing
                text_lower = text.lower()
                is_summary = (
                    ("user asked" in text_lower and "replied" in text_lower)
                    or "conversation summary" in text_lower
                    or text_lower.lstrip().startswith("previously,")
                )
                if is_summary:
                    logging.info(f"Skipping summary-style response: {text[:200]}")
                    return None
                # Strip XML-style tags the model may wrap around the response
                text = re.sub(r"</?[a-zA-Z_]+>", "", text).strip()
                if not text:
                    return None

                fn_name = self.function_schemas[0]["function"]["name"]
                logging.info(
                    f"No tool calls found, wrapping text as {fn_name}: " f"{text[:200]}"
                )
                action = Action(
                    type=fn_name,
                    value=json.dumps({"response": text}, ensure_ascii=False),
                )
                result = CortexOutputModel(actions=[action])
                return T.cast(R, result)

            logging.warning(f"Qwen returned no tool calls and no usable text: {text}")
            return None
        except Exception as e:
            logging.error(f"Qwen LLM error: {e}")
            return None
