import logging
import typing as T
from enum import Enum

from pydantic import BaseModel, Field

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM

R = T.TypeVar("R", bound=BaseModel)


class GeminiModel(str, Enum):
    """Available Gemini models."""

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"


class GeminiConfig(LLMConfig):
    """Gemini-specific configuration with model enum."""

    base_url: T.Optional[str] = Field(
        default="https://api.openmind.org/api/core/gemini",
        description="Base URL for the Gemini API endpoint",
    )
    model: T.Optional[T.Union[GeminiModel, str]] = Field(
        default=GeminiModel.GEMINI_2_5_FLASH,
        description="Gemini model to use",
    )


class GeminiLLM(OpenAICompatibleLLM[R]):
    """Google Gemini LLM implementation using OpenAI-compatible API."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    PROVIDER_NAME = "Gemini"
    LOG_LEVEL = logging.DEBUG
