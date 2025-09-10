import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class XAILLM(LLM[R]):
    """
    XAI LLM implementation using OpenAI-compatible API.

    Handles authentication and response parsing for XAI endpoints.

    Parameters
    ----------
    output_model : Type[R]
        A Pydantic BaseModel subclass defining the expected response structure.
    config : LLMConfig
        Configuration object containing API settings. If not provided, defaults
        will be used.
    """

    def __init__(self, output_model: T.Type[R], config: LLMConfig = LLMConfig()):
        """
        Initialize the DeepSeek LLM instance.

        Parameters
        ----------
        output_model : Type[R]
            Pydantic model class for response validation.
        config : LLMConfig, optional
            Configuration settings for the LLM.
        """
        super().__init__(output_model, config)

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

    @LLMHistoryManager.update_history()
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]]) -> R | None:
        """
        Execute LLM query and parse response

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

            messages = [
                {
                    "role": "system",
                    "content": f"Respond with valid JSON matching this schema: {self._output_model.model_json_schema()}",
                },
                *messages,
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            message_content = response.choices[0].message.content
            self.io_provider.llm_end_time = time.time()

            try:
                parsed_response = self._output_model.model_validate_json(
                    message_content
                )
                logging.debug(f"XAI LLM output: {parsed_response}")
                return parsed_response
            except Exception as e:
                logging.error(f"Error parsing XAI response: {e}")
                return None
        except Exception as e:
            logging.error(f"XAI API error: {e}")
            return None
