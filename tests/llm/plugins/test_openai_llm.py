from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.openai_llm import OpenAILLM


class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    return LLMConfig(base_url="test_url/", api_key="test_key", model="test_model")


@pytest.fixture
def mock_response():
    """Fixture providing a valid mock API response"""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content='{"test_field": "success"}', tool_calls=None)
        )
    ]
    return response


@pytest.fixture
def mock_response_with_tool_calls():
    """Fixture providing a mock API response with tool calls"""
    tool_call = MagicMock()
    tool_call.function.name = "test_function"
    tool_call.function.arguments = '{"arg1": "value1"}'

    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"test_field": "success"}', tool_calls=[tool_call]
            )
        )
    ]
    return response


@pytest.fixture
def llm(config):
    return OpenAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    assert llm._client.base_url == config.base_url
    assert llm._client.api_key == config.api_key
    assert llm._config.model == config.model


@pytest.mark.asyncio
async def test_init_empty_key():
    config = LLMConfig(base_url="test_url")
    with pytest.raises(ValueError, match="config file missing api_key"):
        OpenAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_ask_success(llm, mock_response):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(return_value=mock_response),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_with_tool_calls(llm, mock_response_with_tool_calls):
    """Test successful API request with tool calls"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response_with_tool_calls),
        )

        result = await llm.ask("test prompt")
        assert isinstance(result, CortexOutputModel)
        assert result.actions == [Action(type="test_function", value="value1")]


@pytest.mark.asyncio
async def test_ask_invalid_json(llm):
    invalid_response = MagicMock()
    invalid_response.choices = [MagicMock(message=MagicMock(content="invalid"))]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(return_value=invalid_response),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(side_effect=Exception("API error")),
        )

        result = await llm.ask("test prompt")
        assert result is None
