"""
Tests for LLM retry logic for network errors.

This module tests the retry functionality added to LLM plugins to handle
network interruptions gracefully (Bug #882).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.openai_llm import OpenAILLM, RETRYABLE_EXCEPTIONS


@pytest.fixture
def config():
    return LLMConfig(
        base_url="test_url/",
        api_key="test_key",
        model="test_model",
        max_retries=2,  # Lower for faster tests
        retry_delay=0.01,  # Very short for tests
    )


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
            "llm.plugins.openai_llm.AvatarLLMState.trigger_thinking", mock_decorator
        ),
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
async def test_retry_on_connection_error(llm, mock_response_with_tool_calls):
    """Test that connection errors trigger retry with eventual success"""
    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:  # Fail on first call, succeed on second
            raise openai.APIConnectionError(request=MagicMock())
        return mock_response_with_tool_calls

    with pytest.MonkeyPatch.context() as m:
        m.setattr(llm._client.chat.completions, "create", mock_create)

        result = await llm.ask("test prompt")

        assert call_count == 2  # Should have retried once
        assert isinstance(result, CortexOutputModel)
        assert result.actions == [Action(type="test_function", value="value1")]


@pytest.mark.asyncio
async def test_retry_on_timeout_error(llm, mock_response_with_tool_calls):
    """Test that timeout errors trigger retry"""
    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise openai.APITimeoutError(request=MagicMock())
        return mock_response_with_tool_calls

    with pytest.MonkeyPatch.context() as m:
        m.setattr(llm._client.chat.completions, "create", mock_create)

        result = await llm.ask("test prompt")

        assert call_count == 2
        assert isinstance(result, CortexOutputModel)


@pytest.mark.asyncio
async def test_max_retries_exhausted(llm):
    """Test that all retries exhausted returns None"""
    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise openai.APIConnectionError(request=MagicMock())

    with pytest.MonkeyPatch.context() as m:
        m.setattr(llm._client.chat.completions, "create", mock_create)

        result = await llm.ask("test prompt")

        # max_retries=2 means 3 total attempts (initial + 2 retries)
        assert call_count == 3
        assert result is None


@pytest.mark.asyncio
async def test_no_retry_on_auth_error(llm):
    """Test that authentication errors do NOT trigger retry"""
    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body=None,
        )

    with pytest.MonkeyPatch.context() as m:
        m.setattr(llm._client.chat.completions, "create", mock_create)

        result = await llm.ask("test prompt")

        assert call_count == 1  # Should NOT retry
        assert result is None


@pytest.mark.asyncio
async def test_retryable_exceptions_defined():
    """Test that RETRYABLE_EXCEPTIONS contains expected exception types"""
    assert openai.APIConnectionError in RETRYABLE_EXCEPTIONS
    assert openai.APITimeoutError in RETRYABLE_EXCEPTIONS
    assert ConnectionError in RETRYABLE_EXCEPTIONS
    assert TimeoutError in RETRYABLE_EXCEPTIONS
    assert OSError in RETRYABLE_EXCEPTIONS


@pytest.mark.asyncio
async def test_config_retry_settings():
    """Test that retry settings are properly read from config"""
    config = LLMConfig(
        api_key="test_key",
        max_retries=5,
        retry_delay=2.0,
    )

    assert config.max_retries == 5
    assert config.retry_delay == 2.0


@pytest.mark.asyncio
async def test_default_retry_settings():
    """Test default retry settings"""
    config = LLMConfig(api_key="test_key")

    assert config.max_retries == 3
    assert config.retry_delay == 1.0
