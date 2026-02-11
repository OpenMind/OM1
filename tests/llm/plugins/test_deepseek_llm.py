from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm.output_model import Action, CortexOutputModel
from llm.plugins.deepseek_llm import DeepSeekConfig, DeepSeekLLM


class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    return DeepSeekConfig(base_url="test_url/", api_key="test_key", model="test_model")


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
    return DeepSeekLLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    """Test initialization with provided configuration"""
    assert llm._client.base_url == config.base_url
    assert llm._client.api_key == config.api_key
    assert llm._config.model == config.model


@pytest.mark.asyncio
async def test_init_empty_key():
    """Test fallback API key when no credentials provided"""
    config = DeepSeekConfig(base_url="test_url")
    with pytest.raises(ValueError, match="config file missing api_key"):
        DeepSeekLLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_ask_success(llm, mock_response):
    """Test successful API request and response parsing"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
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
    """Test handling of invalid JSON response"""
    invalid_response = MagicMock()
    invalid_response.choices = [MagicMock(message=MagicMock(content="invalid"))]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=invalid_response),
        )

        result = await llm.ask("test prompt")
        assert result == CortexOutputModel(actions=[])


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    """Test error handling for API exceptions"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(side_effect=Exception("API error")),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_io_provider_timing(llm, mock_response):
    """Test timing metrics collection"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )

        await llm.ask("test prompt")
        assert llm.io_provider.llm_start_time is not None
        assert llm.io_provider.llm_end_time is not None
        assert llm.io_provider.llm_end_time >= llm.io_provider.llm_start_time


@pytest.mark.asyncio
async def test_init_without_model():
    """Test initialization without model defaults to deepseek-chat"""
    config = DeepSeekConfig(api_key="test_key")
    llm = DeepSeekLLM(config, available_actions=None)
    assert llm._config.model == "deepseek-chat"


@pytest.mark.asyncio
async def test_ask_empty_choices(llm):
    """Test handling when API returns empty choices"""
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
async def test_ask_with_multiple_tool_calls(llm):
    """Test handling multiple tool calls in one response"""
    tool_call_1 = MagicMock()
    tool_call_1.function.name = "function_1"
    tool_call_1.function.arguments = '{"arg1": "value1"}'
    
    tool_call_2 = MagicMock()
    tool_call_2.function.name = "function_2"
    tool_call_2.function.arguments = '{"arg2": "value2"}'
    
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(tool_calls=[tool_call_1, tool_call_2])
        )
    ]
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.chat.completions,
            "create",
            AsyncMock(return_value=response),
        )
        result = await llm.ask("test prompt")
        assert isinstance(result, CortexOutputModel)
        assert len(result.actions) == 2


@pytest.mark.asyncio
async def test_ask_timeout_configuration(llm, mock_response):
    """Test that timeout configuration is passed to API"""
    llm._config.timeout = 30
    
    with pytest.MonkeyPatch.context() as m:
        mock_create = AsyncMock(return_value=mock_response)
        m.setattr(llm._client.chat.completions, "create", mock_create)
        
        await llm.ask("test prompt")
        
        assert mock_create.call_args.kwargs['timeout'] == 30


@pytest.mark.asyncio
async def test_ask_with_available_actions(config):
    """Test initialization with available actions"""
    actions = [MagicMock(name="test_action")]
    llm = DeepSeekLLM(config, available_actions=actions)
    assert llm.function_schemas is not None