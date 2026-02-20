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


@pytest.fixture(autouse=True)
def mock_avatar_components():
    """Mock all avatar and IO components to prevent Zenoh session creation"""

    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch(
            "llm.plugins.deepseek_llm.AvatarLLMState.trigger_thinking", mock_decorator
        ),
        patch("llm.plugins.deepseek_llm.AvatarLLMState") as mock_avatar_state,
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
    config = OpenAIConfig(base_url="test_url")
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


def make_response(tool_calls=None, content="{}"):
    """Helper to build a mock completion response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    response = MagicMock()
    response.choices = [mock_choice]
    return response


@pytest.mark.asyncio
async def test_ask_without_messages_only_prompt(llm):
    """When no messages are provided, sent_messages should contain only one user prompt."""
    response = make_response(tool_calls=None)

    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = response
        await llm.ask("only prompt")

        sent_messages = mock_create.call_args.kwargs["messages"]
        assert len(sent_messages) == 1
        assert sent_messages[0] == {"role": "user", "content": "only prompt"}


@pytest.mark.asyncio
async def test_init_model_none_falls_back_to_default():
    """When model is None, it should fall back to 'gpt-4.1-mini'."""
    config = OpenAIConfig(base_url="test_url/", api_key="test_key", model=None)
    llm = OpenAILLM(config, available_actions=None)
    assert llm._config.model == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_init_base_url_none_falls_back_to_default():
    """When base_url is None, the client should use the default OpenMind URL."""
    config = OpenAIConfig(base_url=None, api_key="test_key", model="gpt-4o")
    llm = OpenAILLM(config, available_actions=None)
    assert "openmind.org" in str(llm._client.base_url)


@pytest.mark.asyncio
async def test_ask_multiple_tool_calls(llm):
    """When there are multiple tool calls, function_call_data should include all calls."""

    def make_tool_call(name, args):
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args
        return tc

    tool_calls = [
        make_tool_call("action_one", '{"key": "val1"}'),
        make_tool_call("action_two", '{"key": "val2"}'),
        make_tool_call("action_three", '{"key": "val3"}'),
    ]
    response = make_response(tool_calls=tool_calls)

    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = response
        with patch(
            "llm.plugins.openai_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            mock_convert.return_value = [
                Action(type="action_one", value="val1"),
                Action(type="action_two", value="val2"),
                Action(type="action_three", value="val3"),
            ]
            result = await llm.ask("test")

            # Verify that the function_call_data passed to convert contains three items
            call_args = mock_convert.call_args[0][0]
            assert len(call_args) == 3
            assert call_args[0]["function"]["name"] == "action_one"
            assert call_args[1]["function"]["name"] == "action_two"
            assert call_args[2]["function"]["name"] == "action_three"

            # Verify that the result contains three actions
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 3


@pytest.mark.asyncio
async def test_ask_tool_call_data_format(llm):
    """Verify that each item passed to convert_function_calls_to_actions has the correct data format."""
    tc = MagicMock()
    tc.function.name = "my_action"
    tc.function.arguments = '{"param": 42}'
    response = make_response(tool_calls=[tc])

    with patch.object(
        llm._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = response
        with patch(
            "llm.plugins.openai_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            mock_convert.return_value = []
            await llm.ask("test")

            call_args = mock_convert.call_args[0][0]
            assert len(call_args) == 1
            item = call_args[0]
            assert "function" in item
            assert item["function"]["name"] == "my_action"
            assert item["function"]["arguments"] == '{"param": 42}'


def test_history_manager_update_history_applied(llm):
    """A decorator usually leaves a marker on the function object, or we can verify
    that LLMHistoryManager has been properly initialized.
    """
    assert hasattr(llm, "history_manager")
    assert llm.history_manager is not None
