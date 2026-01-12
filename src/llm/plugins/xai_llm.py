import asyncio
import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from llm.error_handler import handle_llm_api_error, is_retryable_error
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class XAILLM(LLM[R]):
    """
    XAI LLM implementation using OpenAI-compatible API.

    Handles authentication and response parsing for XAI endpoints.

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided.
    """

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the XAI LLM instance.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "grok-4-latest"

        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or "https://api.openmind.org/api/core/xai",
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
            logging.debug(f"XAI LLM input: {prompt}")
            logging.debug(f"XAI LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            # Retry logic for gateway errors (502, 503, 504)
            max_retries = 2
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=self._config.model or "grok-4-latest",
                        messages=T.cast(T.Any, formatted_messages),
                        tools=T.cast(T.Any, self.function_schemas),
                        tool_choice="auto",
                        timeout=self._config.timeout,
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    if is_retryable_error(e) and attempt < max_retries:
                        delay = min(1.0 * (2**attempt), 10.0)
                        logging.warning(
                            f"XAI: Retryable error (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        handle_llm_api_error(e, "XAI")
                        await asyncio.sleep(delay)
                    else:
                        raise

            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()

            if message.tool_calls:
                logging.info(f"Received {len(message.tool_calls)} function calls")
                logging.info(f"Function calls: {message.tool_calls}")

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

                result = CortexOutputModel(actions=actions)
                logging.info(f"XAI LLM function call output: {result}")
                return T.cast(R, result)

            return None
        except Exception as e:
            handle_llm_api_error(e, "XAI")
            return None
