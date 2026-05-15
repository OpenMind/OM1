import logging
import re
import time
import typing as T
from enum import Enum

import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from prometheus import om1_llm_latency, om1_llm_latency_last
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


def _extract_voice_input(prompt: str) -> str:
    """
    Extract voice input from the prompt.

    Parameters
    ----------
    prompt : str
        Full prompt containing INPUT: Voice section.

    Returns
    -------
    str
        Extracted voice input text, or empty string if not found.
    """
    match = re.search(r"Voice:\s*([^\n]+)", prompt)
    if match:
        return match.group(1).strip().strip("\"'")

    return ""


class FunctionGemmaModel(str, Enum):
    """Available FunctionGemma models."""

    MULTILINGUAL = "functiongemma-finetuned-g1-multilingual"


class FunctionGemmaConfig(LLMConfig):
    """FunctionGemma-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="http://localhost:8200/v1",
        description="Base URL for the FunctionGemma API endpoint",
    )
    model: T.Optional[T.Union[FunctionGemmaModel, str]] = Field(
        default=FunctionGemmaModel.MULTILINGUAL,
        description="FunctionGemma model to use",
    )


class FunctionGemmaLLM(LLM[R]):
    """
    An FunctionGemma-based Language Learning Model implementation with function call support.

    This class implements the LLM interface for FunctionGemma's models, handling
    configuration, authentication, and async API communication. It supports both
    traditional JSON structured output and function calling.
    """

    def __init__(
        self,
        config: FunctionGemmaConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the FunctionGemma LLM instance.

        Parameters
        ----------
        config : FunctionGemmaConfig, optional
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = FunctionGemmaModel.MULTILINGUAL

        self.base_url = config.base_url or "http://localhost:8200/v1"
        self._client = openai.AsyncClient(
            base_url=self.base_url,
            api_key=config.api_key,
        )

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(self, prompt: str, messages: T.Optional[T.List[T.Dict[str, str]]] = None) -> T.Optional[R]:
        """
        Send a prompt to the FunctionGemma API and get a structured response.

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
        try:
            logging.info(f"FunctionGemma input: {prompt}")

            llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            # FunctionGemma expects messages in a specific format, so we wrap the prompt accordingly
            voice = _extract_voice_input(prompt)
            message = [{"role": "user", "content": voice}]

            response = await self._client.chat.completions.create(
                model=self._config.model or FunctionGemmaModel.MULTILINGUAL,
                messages=T.cast(T.Any, message),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            if not response.choices:
                logging.warning("FunctionGemma API returned empty choices")
                return None

            message = response.choices[0].message
            latency = time.time() - llm_start_time
            om1_llm_latency.labels(
                model=str(self._config.model or FunctionGemmaModel.MULTILINGUAL),
                endpoint=str(self.base_url),
            ).observe(latency)
            om1_llm_latency_last.labels(
                model=str(self._config.model or FunctionGemmaModel.MULTILINGUAL),
                endpoint=str(self.base_url),
            ).set(latency)

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
            logging.error(f"FunctionGemma API error: {e}")
            return None
