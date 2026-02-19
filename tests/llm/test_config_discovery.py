"""Tests for LLM config class discovery."""

import pytest
from llm import get_llm_config_class, LLMConfig


class TestGetLLMConfigClass:
    """Test suite for get_llm_config_class helper function."""

    def test_get_llm_config_class_returns_config_subclass(self):
        """Test that get_llm_config_class returns a subclass of LLMConfig."""
        config_class = get_llm_config_class("QwenLLM")
        assert issubclass(config_class, LLMConfig)
        assert config_class != LLMConfig  # Should be specific subclass

    def test_get_llm_config_class_for_qwen(self):
        """Test config class discovery for QwenLLM."""
        config_class = get_llm_config_class("QwenLLM")
        assert config_class is not None
        assert config_class.__name__ == "QwenLLMConfig"

    def test_get_llm_config_class_for_openai(self):
        """Test config class discovery for OpenAILLM."""
        config_class = get_llm_config_class("OpenAILLM")
        assert config_class is not None
        assert config_class.__name__ == "OpenAIConfig"

    def test_get_llm_config_class_raises_for_invalid(self):
        """Test that invalid class name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_llm_config_class("NonExistentLLM")

    def test_get_llm_config_class_raises_for_base_llm(self):
        """Test that base LLM class raises ValueError."""
        with pytest.raises(ValueError, match="not a valid LLM subclass"):
            get_llm_config_class("LLM")

    def test_config_class_can_be_instantiated(self):
        """Test that discovered config class can be instantiated."""
        config_class = get_llm_config_class("QwenLLM")
        config = config_class(base_url="http://localhost:8000/v1", api_key="test")
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key == "test"
