import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class NearAIModel(str, Enum):
    """Available NearAI models."""

    QWEN_30B_A3B_INSTRUCT_2507 = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    DEEPSEEK_V3_1 = "deepseek-ai/DeepSeek-V3.1"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_5_2 = "openai/gpt-5.2"
    GLM_4_7 = "zai-org/GLM-4.7"
    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4-5"
    GEMINI_3_PRO = "google/gemini-3-pro"


class NearAIConfig(LLMConfig):
    """NearAI-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/nearai",
        description="Base URL for the NearAI API endpoint",
    )
    model: T.Optional[T.Union[NearAIModel, str]] = Field(
        default=NearAIModel.GPT_OSS_120B,
        description="NearAI model to use",
    )


class NearAILLM(OpenAICompatibleLLM[R]):
    """NearAI LLM implementation using beta.chat.completions.parse API."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    PROVIDER_NAME = "NearAI"
    LOG_LEVEL = logging.INFO

    async def _call_api(self, formatted_messages: T.List[T.Dict[str, str]]) -> T.Any:
        """Use beta.chat.completions.parse instead of chat.completions.create."""
        return await self._client.beta.chat.completions.parse(
            model=self._config.model or self.DEFAULT_MODEL,
            messages=T.cast(T.Any, formatted_messages),
            tools=T.cast(T.Any, self.function_schemas),
            tool_choice="auto",
            timeout=self._config.timeout,
        )
