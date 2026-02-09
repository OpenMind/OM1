import logging
import time
import typing as T
from enum import Enum

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class OpenRouterModel(str, Enum):
    """Available OpenRouter models."""

    ANTHROPIC_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    ANTHROPIC_OPUS_4_5 = "anthropic/claude-opus-4.5"
    ANTHROPIC_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    MOONSHOT_KIMI_K2_5 = "moonshotai/kimi-k2.5"
    MINIMAX_M2_1 = "minimax/minimax-m2.1"
    Z_AI_GLM_4_7 = "z-ai/glm-4.7"
    X_AI_GROK_4_FAST = "x-ai/grok-4-fast"
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/openrouter",
        description="Base URL for the OpenRouter API endpoint",
    )
    model: T.Optional[T.Union[OpenRouterModel, str]] = Field(
        default=OpenRouterModel.ANTHROPIC_SONNET_4_5,
        description="OpenRouter model to use",
    )


class OpenRouter(LLM[R]):
    """
    An OpenRouter-based Language Learning Model implementation with function call support.

    This class implements the LLM interface for OpenRouter's models (Meta and Anthropic), handling
    configuration, authentication, and async API communication. It supports both
    traditional JSON structured output and function calling.
    """

    def __init__(
        self,
        config: OpenRouterConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the OpenRouter instance.

        Parameters
        ----------
        config : OpenRouterConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "meta-llama/llama-3.3-70b-instruct"

        self._client = openai.AsyncClient(
            base_url=config.base_url or "https://api.openmind.org/api/core/openrouter",
            api_key=config.api_key,
        )

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to the OpenRouter API and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries to send to the model.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.info(f"OpenRouter input: {prompt}")
            logging.info(f"OpenRouter messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._config.model or "meta-llama/llama-3.3-70b-instruct",
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            if not response.choices:
                logging.warning("OpenRouter API returned empty choices")
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
                logging.info(f"OpenRouter function call output: {result}")
                return T.cast(R, result)

            return None

        except Exception as e:
            logging.error(f"OpenRouter API error: {e}")
            return None
