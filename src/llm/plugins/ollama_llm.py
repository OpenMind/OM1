import json
import logging
import time
import typing as T

import httpx
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class OllamaLLMConfig(LLMConfig):
    """
    Configuration for Ollama LLM.

    Parameters
    ----------
    base_url : str
        Base URL for Ollama API (default: http://localhost:11434)
    model : str
        Ollama model name (e.g., llama3.2, mistral, phi3, llava)
    temperature : float
        Sampling temperature (0.0 - 1.0)
    num_ctx : int
        Context window size
    """

    base_url: T.Optional[str] = Field(
        default="http://localhost:11434", description="Base URL for Ollama API"
    )
    model: T.Optional[str] = Field(default="llama3.2", description="Ollama model name")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    num_ctx: int = Field(default=4096, description="Context window size")
    timeout: T.Optional[int] = Field(
        default=120,
        description="Request timeout in seconds (longer for local inference)",
    )


class OllamaLLM(LLM[R]):
    """
    Ollama-based Language Learning Model implementation.

    This class implements the LLM interface for local Ollama models,
    providing privacy-focused, cost-free, offline-capable inference.

    Parameters
    ----------
    config : OllamaLLMConfig
        Configuration object containing Ollama settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation.
    """

    def __init__(
        self,
        config: OllamaLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the Ollama LLM instance.

        Parameters
        ----------
        config : OllamaLLMConfig
            Configuration settings for Ollama.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: OllamaLLMConfig = config

        self._base_url = (self._config.base_url or "http://localhost:11434").strip("/")
        self._chat_url = f"{self._base_url}/api/chat"

        self._client = httpx.AsyncClient(timeout=config.timeout)

        # Initialize history manager
        self.history_manager = LLMHistoryManager(
            self._config, self._client  # type: ignore
        )

        logging.info(f"OllamaLLM initialized with model: {config.model}")
        logging.info(f"Ollama endpoint: {self._chat_url}")

    def _convert_tools_to_ollama_format(self) -> T.List[T.Dict]:
        """
        Convert function schemas to Ollama's tool format.

        Returns
        -------
        list
            List of tools in Ollama format
        """
        if not self.function_schemas:
            return []

        ollama_tools = []
        for schema in self.function_schemas:
            tool = {
                "type": "function",
                "function": {
                    "name": schema["function"]["name"],
                    "description": schema["function"].get("description", ""),
                    "parameters": schema["function"].get("parameters", {}),
                },
            }
            ollama_tools.append(tool)

        return ollama_tools

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to Ollama and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries for conversation history.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.info(f"Ollama input: {prompt}")
            logging.debug(f"Ollama messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self._config.model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": self._config.temperature,
                    "num_ctx": self._config.num_ctx,
                },
            }

            tools = self._convert_tools_to_ollama_format()
            if tools:
                payload["tools"] = tools

            logging.debug(f"Ollama request payload: {json.dumps(payload, indent=2)}")

            response = await self._client.post(
                self._chat_url,
                json=payload,
            )

            if response.status_code != 200:
                logging.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            self.io_provider.llm_end_time = time.time()

            logging.debug(f"Ollama response: {json.dumps(result, indent=2)}")

            message = result.get("message", {})

            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                logging.info(f"Received {len(tool_calls)} function calls from Ollama")
                logging.info(f"Function calls: {tool_calls}")

                function_call_data = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    function_call_data.append(
                        {
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": (
                                    json.dumps(func.get("arguments", {}))
                                    if isinstance(func.get("arguments"), dict)
                                    else func.get("arguments", "{}")
                                ),
                            }
                        }
                    )

                actions = convert_function_calls_to_actions(function_call_data)
                result_model = CortexOutputModel(actions=actions)
                return T.cast(R, result_model)

            return None

        except httpx.ConnectError as e:
            logging.error(
                f"Cannot connect to Ollama at {self._base_url}. Is Ollama running?"
            )
            logging.error("Start Ollama with: ollama serve")
            logging.error(f"Error: {e}")
            return None
        except httpx.TimeoutException as e:
            logging.error(f"Ollama request timed out after {self._config.timeout}s")
            logging.error("Try increasing timeout or using a smaller model")
            logging.error(f"Error: {e}")
            return None
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
