from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from llm.plugins.openrouter import OpenRouter, OpenRouterConfig


@pytest.fixture
def config():
    return OpenRouterConfig(
        base_url="test_url/", api_key="test_key", model="test_model"
    )


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
    return OpenRouter(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    assert llm._client.base_url == config.base_url
    assert llm._client.api_key == config.api_key
    assert llm._config.model == config.model


@pytest.mark.asyncio
async def test_init_empty_key():
    config = OpenRouterConfig(base_url="test_url")
    with pytest.raises(ValueError, match="config file missing api_key"):
        OpenRouter(config, available_actions=None)


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


@pytest.mark.asyncio
async def test_ask_api_status_error(llm):
    """Test error handling for HTTP status errors (e.g. 502 Bad Gateway)"""
    import openai

    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.headers = {}
    error = openai.APIStatusError(
        message="Bad Gateway",
        response=mock_response,
        body=None,
    )
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(side_effect=error),
        )
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_init_no_model(config):
    """Test that missing model defaults to llama"""
    config.model = None
    llm = OpenRouter(config, available_actions=None)
    assert llm._config.model == "meta-llama/llama-3.3-70b-instruct"


@pytest.mark.asyncio
async def test_ask_messages_none_branch(llm, mock_response_with_tool_calls):
    """Test messages=None branch by calling the unwrapped function directly"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response_with_tool_calls),
        )
        # Call __wrapped__ to bypass decorators and hit messages=None branch
        result = await llm.ask.__wrapped__.__wrapped__(
            llm, "test prompt", messages=None
        )
        assert isinstance(result, CortexOutputModel)


@pytest.mark.asyncio
async def test_ask_empty_choices(llm):
    """Test ask returns None when API returns empty choices"""
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


@pytest.mark.asyncio
async def test_ask_no_tool_calls_returns_none(llm):
    """Test ask returns None when response has no tool calls"""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(tool_calls=None))]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=response),
        )
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_api_connection_error_branch(llm):
    """Test APIConnectionError branch is hit"""
    import openai

    error = openai.APIConnectionError(request=MagicMock())
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(side_effect=error),
        )
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_unexpected_error_branch(llm):
    """Test generic Exception branch is hit"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(side_effect=RuntimeError("something broke")),
        )
        result = await llm.ask("test prompt")
        assert result is None
