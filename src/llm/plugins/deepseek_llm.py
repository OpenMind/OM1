import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class DeepSeekLLM(LLM[R]):
    """
    An DeepSeek-based Language Learning Model implementation.

    This class implements the LLM interface for DeepSeek's conversation models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings. If not provided, defaults
        will be used.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided,
    """

    def __init__(
        self,
        config: LLMConfig = LLMConfig(),
        available_actions: T.Optional[T.List] = None,
    ):
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
            self._config.model = "deepseek-chat"

        self._client = openai.OpenAI(
            base_url=config.base_url or "https://api.openmind.org/api/core/deepseek",
            api_key=config.api_key,
        )

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

    @LLMHistoryManager.update_history()
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]]) -> R | None:
        """
        Send a prompt to the DeepSeek API and get a structured response.

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
            logging.debug(f"DeepSeek LLM input: {prompt}")
            logging.debug(f"DeepSeek LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            messages = [
                {
                    "role": "system",
                    "content": f"You must respond with valid JSON matching this schema: {self._output_model.model_json_schema()}",
                },
                *messages,
                {"role": "user", "content": prompt},
            ]

            parsed_response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            message_content = parsed_response.choices[0].message.content
            self.io_provider.llm_end_time = time.time()

            try:
                parsed_response = self._output_model.model_validate_json(
                    message_content
                )
                logging.debug(f"DeepSeek LLM output: {parsed_response}")
                return parsed_response
            except Exception as e:
                logging.error(f"Error parsing DeepSeek response: {e}")
                return None
        except Exception as e:
            logging.error(f"DeepSeek API error: {e}")
            return None
