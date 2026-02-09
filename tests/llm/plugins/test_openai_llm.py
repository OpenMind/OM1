from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm.output_model import Action, CortexOutputModel
from llm.plugins.openai_llm import OpenAIConfig, OpenAILLM


class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    return OpenAIConfig(base_url="test_url/", api_key="test_key", model="test_model")


@pytest.fixture
def mock_completion_response_with_tool_calls():
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "test_function"
    mock_tool_call.function.arguments = '{"arg1": "value1"}'
    mock_message = MagicMock()
    mock_message.content = '{"test_field": "success"}'
    mock_message.tool_calls = [mock_tool_call]
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    response = MagicMock()
    response.choices = [mock_choice]
    return response


@pytest.fixture
def mock_completion_response_without_tool_calls():
    mock_message = MagicMock()
    mock_message.content = '{"test_field": "success"}'
    mock_message.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    response = MagicMock()
    response.choices = [mock_choice]
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
        patch("llm.plugins.openai_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.openai_llm.AvatarLLMState") as mock_avatar_state,
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
    return OpenAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    assert llm._client.base_url == config.base_url
    assert llm._client.api_key == config.api_key
    assert llm._config.model == config.model


@pytest.mark.asyncio
async def test_init_empty_key():
    config = OpenAIConfig(base_url="test_url", api_key="")
    with pytest.raises(ValueError, match="config file missing api_key"):
        OpenAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_ask_success_with_tool_calls(
    llm, mock_completion_response_with_tool_calls
):
    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_completion_response_with_tool_calls
        with patch(
            "llm.plugins.openai_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            expected_action = Action(type="test_function", value="value1")
            mock_convert.return_value = [expected_action]
            result = await llm.ask("test prompt")
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == llm._config.model
            assert "messages" in call_kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == "auto"
            mock_convert.assert_called_once()
            assert isinstance(result, CortexOutputModel)
            assert result.actions == [expected_action]


@pytest.mark.asyncio
async def test_ask_success_without_tool_calls(
    llm, mock_completion_response_without_tool_calls
):
    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_completion_response_without_tool_calls
        with patch(
            "llm.plugins.openai_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            result = await llm.ask("test prompt")
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == llm._config.model
            assert "messages" in call_kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == "auto"
            mock_convert.assert_not_called()
            assert result is None


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = Exception("API error")
        result = await llm.ask("test prompt")
        mock_create.assert_called_once()
        assert result is None


@pytest.mark.asyncio
async def test_ask_empty_choices(llm):
    mock_response_empty_choices = MagicMock()
    mock_response_empty_choices.choices = []
    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response_empty_choices
        result = await llm.ask("test prompt")
        mock_create.assert_called_once()
        assert result is None
