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
from providers.vector_memory_provider import VectorMemoryProvider

R = T.TypeVar("R", bound=BaseModel)


class OpenAILLM(LLM[R]):
    """
    An OpenAI-based Language Learning Model implementation with function call support.

    This class implements the LLM interface for OpenAI's GPT models, handling
    configuration, authentication, and async API communication. It supports both
    traditional JSON structured output and function calling.

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
        Initialize the OpenAI LLM instance.

        Parameters
        ----------
        config : LLMConfig, optional
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "gpt-4.1-mini"

        self._client = openai.AsyncClient(
            base_url=config.base_url or "https://api.openmind.org/api/core/openai",
            api_key=config.api_key,
        )

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

        # Initialize vector memory if configured
        self.vector_memory = None
        if hasattr(config, "vector_memory") and config.vector_memory:
            try:
                agent_name = getattr(config, "agent_name", "Robot")
                self.vector_memory = VectorMemoryProvider(
                    config.vector_memory, agent_name=agent_name
                )
                if self.vector_memory.enabled:
                    logging.info("Vector Memory enabled for OpenAI LLM")
            except Exception as e:
                logging.error(f"Failed to initialize Vector Memory: {e}")
                self.vector_memory = None

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to the OpenAI API and get a structured response.

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
            # Retrieve relevant long-term memories before calling LLM
            memory_context = ""
            if self.vector_memory and self.vector_memory.enabled:
                memory_context = await self.vector_memory.get_enriched_context(
                    current_user_message=prompt
                )

            logging.info(f"OpenAI input: {prompt}")
            logging.info(f"OpenAI messages: {messages}")
            if memory_context:
                memory_count = memory_context.count("[Memory")
                logging.info(
                    f"Retrieved {memory_count} relevant memories from vector storage"
                )

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]

            # If we have memory context, inject it as a system message before user prompt
            if memory_context:
                formatted_messages.append({"role": "system", "content": memory_context})

            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._config.model or "gpt-5",
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

                # Store this interaction in vector memory after getting response
                if self.vector_memory and self.vector_memory.enabled:
                    robot_response = self._extract_response_from_actions(actions)
                    await self.vector_memory.store_conversation_turn(
                        user_message=prompt, robot_response=robot_response
                    )

                return T.cast(R, result)

            # If no tool calls but has text response, still store it
            if message.content and self.vector_memory and self.vector_memory.enabled:
                await self.vector_memory.store_conversation_turn(
                    user_message=prompt, robot_response=message.content
                )

            return None

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return None

    def _extract_response_from_actions(self, actions: T.List) -> str:
        """
        Extract readable response from actions for memory storage

        Args:
            actions: List of actions from LLM

        Returns:
            String representation of robot's response
        """
        responses = []
        for action in actions:
            if hasattr(action, "name") and hasattr(action, "parameters"):
                # Extract speak action content
                if action.name == "speak" and "text" in action.parameters:
                    responses.append(action.parameters["text"])
                # Extract emotion
                elif action.name == "face" and "emotion" in action.parameters:
                    responses.append(f"[emotion: {action.parameters['emotion']}]")
                # Extract movement
                elif action.name == "move" and "motion" in action.parameters:
                    responses.append(f"[move: {action.parameters['motion']}]")

        return " ".join(responses) if responses else "[action performed]"
