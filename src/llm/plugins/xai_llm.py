import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class XAIModel(str, Enum):
    """Available XAI models."""

    GROK_2_LATEST = "grok-2-latest"
    GROK_3_BETA = "grok-3-beta"
    GROK_4_LATEST = "grok-4-latest"
    GROK_4 = "grok-4"


class XAIConfig(LLMConfig):
    """XAI-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/xai",
        description="Base URL for the XAI API endpoint",
    )
    model: T.Optional[T.Union[XAIModel, str]] = Field(
        default=XAIModel.GROK_4_LATEST,
        description="XAI model to use",
    )


class XAILLM(OpenAICompatibleLLM[R]):
    """XAI LLM implementation using OpenAI-compatible API."""

    DEFAULT_MODEL = "grok-4-latest"
    PROVIDER_NAME = "XAI"
    LOG_LEVEL = logging.DEBUG
