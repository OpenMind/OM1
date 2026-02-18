import logging
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


class OpenAICompatibleLLM(LLM[R]):
    """Base class for OpenAI-API-compatible LLM implementations.

    Subclasses only need to define class attributes (DEFAULT_MODEL, PROVIDER_NAME,
    LOG_LEVEL) and their Config/Model enums. Override _call_api() for providers
    that use a different API method (e.g. beta.chat.completions.parse).
    """

    DEFAULT_MODEL: str = "gpt-4.1-mini"
    PROVIDER_NAME: str = "OpenAI"
    LOG_LEVEL: int = logging.INFO

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = self.DEFAULT_MODEL

        self._client = openai.AsyncClient(
            base_url=config.base_url,
            api_key=config.api_key,
        )

        self.history_manager = LLMHistoryManager(self._config, self._client)

    async def _call_api(self, formatted_messages: T.List[T.Dict[str, str]]) -> T.Any:
        """Call the chat completions API. Override for different API methods."""
        return await self._client.chat.completions.create(
            model=self._config.model or self.DEFAULT_MODEL,
            messages=T.cast(T.Any, formatted_messages),
            tools=T.cast(T.Any, self.function_schemas),
            tool_choice="auto",
            timeout=self._config.timeout,
        )

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.Optional[T.List[T.Dict[str, str]]] = None
    ) -> T.Optional[R]:
        """Send a prompt to the LLM API and return parsed tool call actions."""
        if messages is None:
            messages = []
        try:
            logging.log(self.LOG_LEVEL, f"{self.PROVIDER_NAME} input: {prompt}")
            logging.log(self.LOG_LEVEL, f"{self.PROVIDER_NAME} messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._call_api(formatted_messages)

            if not response.choices:
                logging.warning(f"{self.PROVIDER_NAME} API returned empty choices")
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
                logging.info(f"{self.PROVIDER_NAME} function call output: {result}")
                return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"{self.PROVIDER_NAME} API error: {e}")
            return None
