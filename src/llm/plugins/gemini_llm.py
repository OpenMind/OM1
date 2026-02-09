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


class GeminiModel(str, Enum):
    """Available Gemini models."""

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_FLASH = "gemini-3-flash"


class GeminiConfig(LLMConfig):
    """Gemini-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/gemini",
        description="Base URL for the Gemini API endpoint",
    )
    model: T.Optional[T.Union[GeminiModel, str]] = Field(
        default=GeminiModel.GEMINI_2_5_FLASH,
        description="Gemini model to use",
    )


class GeminiLLM(LLM[R]):
    """
    Google Gemini LLM implementation using OpenAI-compatible API.

    Handles authentication and response parsing for Gemini endpoints.
    """

    def __init__(
        self,
        config: GeminiConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the Gemini LLM instance.

        Parameters
        ----------
        config : GeminiConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "gemini-2.5-flash"

        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or "https://api.openmind.org/api/core/gemini",
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
        Execute LLM query and parse response.

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
            logging.debug(f"Gemini LLM input: {prompt}")
            logging.debug(f"Gemini LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._config.model or "gemini-2.0-flash-exp",
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            if not response.choices:
                logging.warning("Gemini API returned empty choices")
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
                logging.info(f"Gemini LLM function call output: {result}")
                return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return None
