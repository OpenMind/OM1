"""LiteLLM plugin for OM1.

Routes to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, Bedrock,
Ollama, etc.) via the litellm SDK. No proxy server needed.

Model strings use the ``provider/model`` format, e.g.
``anthropic/claude-sonnet-4-20250514``, ``azure/gpt-4o``,
``bedrock/anthropic.claude-3-haiku``, ``openai/gpt-4o``.

See https://docs.litellm.ai/docs/providers for all supported models.
"""

import logging
import time
import typing as T

from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class LiteLLMConfig(LLMConfig):
    """LiteLLM-specific configuration."""

    base_url: T.Optional[str] = Field(
        default=None,
        description="Optional base URL override for the LLM API endpoint",
    )
    model: T.Optional[str] = Field(
        default="openai/gpt-4o",
        description="LiteLLM model string (e.g. anthropic/claude-sonnet-4-20250514)",
    )


class LiteLLM(LLM[R]):
    """
    A LiteLLM-based Language Learning Model implementation.

    Routes to 100+ LLM providers through the litellm SDK using
    ``litellm.acompletion()``. Supports function calling via the same
    tool_calls interface as OpenAI.
    """

    def __init__(
        self,
        config: LiteLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        super().__init__(config, available_actions)

        if not config.model:
            self._config.model = "openai/gpt-4o"

        try:
            import litellm as _litellm  # noqa: F401
        except ImportError:
            raise ImportError("litellm is required for this plugin. " "Install with: pip install litellm")

        import openai

        self._openai_client = openai.AsyncClient(
            api_key=config.api_key or "unused",
            base_url=config.base_url or "https://api.openai.com/v1",
        )

        self.history_manager = LLMHistoryManager(self._config, self._openai_client)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self,
        prompt: str,
        messages: T.Optional[T.List[T.Dict[str, str]]] = None,
    ) -> T.Optional[R]:
        """
        Send a prompt to the LLM via litellm and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]], optional
            List of message dictionaries to send to the model.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        import litellm as _litellm

        if messages is None:
            messages = []
        try:
            logging.info(f"LiteLLM input: {prompt}")
            logging.info(f"LiteLLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            params: T.Dict[str, T.Any] = {
                "model": self._config.model or "openai/gpt-4o",
                "messages": formatted_messages,
                "drop_params": True,
                "timeout": self._config.timeout,
            }

            if self._config.api_key:
                params["api_key"] = self._config.api_key
            if self._config.base_url:
                params["api_base"] = self._config.base_url

            if self.function_schemas:
                params["tools"] = self.function_schemas
                params["tool_choice"] = "auto"

            response = await _litellm.acompletion(**params)

            if not response.choices:
                logging.warning("LiteLLM API returned empty choices")
                return None

            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()

            if message.tool_calls:
                logging.info(f"Received {len(message.tool_calls)} function calls")
                logging.info(f"Function calls: {message.tool_calls}")

                function_call_data = [
                    {
                        "function": {
                            "name": getattr(tc, "function").name,
                            "arguments": getattr(tc, "function").arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

                actions = convert_function_calls_to_actions(function_call_data)

                result = CortexOutputModel(actions=actions)
                return T.cast(R, result)

            return None

        except Exception as e:
            logging.error(f"LiteLLM API error: {e}")
            return None
