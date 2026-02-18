import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class OpenRouterModel(str, Enum):
    """Available OpenRouter models."""

    ANTHROPIC_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    ANTHROPIC_OPUS_4_5 = "anthropic/claude-opus-4.5"
    ANTHROPIC_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    MOONSHOT_KIMI_K2_5 = "moonshotai/kimi-k2.5"
    MINIMAX_M2_1 = "minimax/minimax-m2.1"
    Z_AI_GLM_4_7 = "z-ai/glm-4.7"
    X_AI_GROK_4_FAST = "x-ai/grok-4-fast"
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/openrouter",
        description="Base URL for the OpenRouter API endpoint",
    )
    model: T.Optional[T.Union[OpenRouterModel, str]] = Field(
        default=OpenRouterModel.ANTHROPIC_SONNET_4_5,
        description="OpenRouter model to use",
    )


class OpenRouter(OpenAICompatibleLLM[R]):
    """OpenRouter LLM implementation using OpenAI-compatible API."""

    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
    PROVIDER_NAME = "OpenRouter"
    LOG_LEVEL = logging.INFO
