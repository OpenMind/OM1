import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class DeepSeekModel(str, Enum):
    """Available DeepSeek models."""

    DEEPSEEK_CHAT = "deepseek-chat"


class DeepSeekConfig(LLMConfig):
    """DeepSeek-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/deepseek",
        description="Base URL for the DeepSeek API endpoint",
    )
    model: T.Optional[T.Union[DeepSeekModel, str]] = Field(
        default=DeepSeekModel.DEEPSEEK_CHAT,
        description="DeepSeek model to use",
    )


class DeepSeekLLM(OpenAICompatibleLLM[R]):
    """DeepSeek LLM implementation using OpenAI-compatible API."""

    DEFAULT_MODEL = "deepseek-chat"
    PROVIDER_NAME = "DeepSeek"
    LOG_LEVEL = logging.DEBUG
