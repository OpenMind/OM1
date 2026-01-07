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


class NearAILLM(LLM[R]):
    """
    An NearAI-based Language Learning Model implementation.

    This class implements the LLM interface for Near AI's open-source models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided,
        the LLM will use function calls instead of structured JSON output.
    """

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the NearAI LLM instance.

        Parameters
        ----------
        config : LLMConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "qwen3-30b-a3b-instruct-2507"

        self._client = openai.AsyncClient(
            base_url=config.base_url or "https://api.openmind.org/api/core/nearai",
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
        Send a prompt to the NearAI API and get a structured response.

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
            logging.info(f"NearAI LLM input: {prompt}")
            logging.info(f"NearAI LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.beta.chat.completions.parse(
                model=self._config.model or "qwen3-30b-a3b-instruct-2507",
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

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
                logging.info(f"OpenAI LLM function call output: {result}")
                return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"NearAI API error: {e}")
            return None
