import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class OpenAIModel(str, Enum):
    """Available OpenAI models."""

    GPT_4_O = "gpt-4o"
    GPT_4_O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_1 = "gpt-5.1"
    GPT_5_2 = "gpt-5.2"


class OpenAIConfig(LLMConfig):
    """OpenAI-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/openai",
        description="Base URL for the OpenAI API endpoint",
    )
    model: T.Optional[T.Union[OpenAIModel, str]] = Field(
        default=OpenAIModel.GPT_4_1_MINI,
        description="OpenAI model to use",
    )


class OpenAILLM(OpenAICompatibleLLM[R]):
    """OpenAI LLM implementation using OpenAI-compatible API."""

    DEFAULT_MODEL = "gpt-4.1-mini"
    PROVIDER_NAME = "OpenAI"
    LOG_LEVEL = logging.INFO
