from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.openai_compatible import OpenAICompatibleLLM
from llm.output_model import Action, CortexOutputModel


class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    return LLMConfig(
        base_url="https://test.example.com/",
        api_key="test_key",
        model="test_model",
    )


@pytest.fixture
def mock_response_with_tool_calls():
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
def mock_response_without_tool_calls():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content='{"test_field": "success"}', tool_calls=None)
        )
    ]
    return response


@pytest.fixture(autouse=True)
def mock_avatar_components():
    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch("llm.openai_compatible.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.openai_compatible.AvatarLLMState") as mock_avatar_state,
        patch("providers.avatar_provider.AvatarProvider") as mock_avatar_provider,
        patch(
            "providers.avatar_llm_state_provider.AvatarProvider"
        ) as mock_avatar_llm_state_provider,
    ):
        mock_avatar_state._instance = None
        mock_avatar_state._lock = None

        mock_provider_instance = MagicMock()
        mock_provider_instance.running = False
        mock_provider_instance.session = None
        mock_provider_instance.stop = MagicMock()
        mock_avatar_provider.return_value = mock_provider_instance
        mock_avatar_llm_state_provider.return_value = mock_provider_instance

        yield


@pytest.fixture
def llm(config):
    return OpenAICompatibleLLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_ask_with_tool_calls(llm, mock_response_with_tool_calls):
    """Test that tool calls are parsed into CortexOutputModel."""
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
async def test_ask_without_tool_calls(llm, mock_response_without_tool_calls):
    """Test that no tool calls returns None."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response_without_tool_calls),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    """Test that API errors return None."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(side_effect=Exception("API error")),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_empty_choices(llm):
    """Test that empty choices returns None."""
    empty_response = MagicMock()
    empty_response.choices = []

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=empty_response),
        )

        result = await llm.ask("test prompt")
        assert result is None


def test_missing_api_key():
    """Test that missing API key raises ValueError."""
    config = LLMConfig(base_url="https://test.example.com/")
    with pytest.raises(ValueError, match="config file missing api_key"):
        OpenAICompatibleLLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_call_api_method(llm, mock_response_with_tool_calls):
    """Test that _call_api is called during ask()."""
    with patch.object(llm, "_call_api", new_callable=AsyncMock) as mock_call_api:
        mock_call_api.return_value = mock_response_with_tool_calls

        result = await llm.ask("test prompt")
        mock_call_api.assert_called_once()
        assert isinstance(result, CortexOutputModel)
