from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm.output_model import Action, CortexOutputModel
from llm.plugins.near_ai_llm import NearAIConfig, NearAILLM


class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    """Fixture providing a basic LLM configuration."""
    return NearAIConfig(
        base_url="https://api.test.nearai.com/",
        api_key="test_api_key",
        model="test-model",
    )


@pytest.fixture
def mock_response():
    """Fixture providing a valid mock API response without tool calls."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content='{"test_field": "success"}', tool_calls=None)
        )
    ]
    return response


@pytest.fixture
def mock_response_with_tool_calls():
    """Fixture providing a mock API response with tool calls."""
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
def mock_response_with_multiple_tool_calls():
    """Fixture providing a mock API response with multiple tool calls."""
    tool_call_1 = MagicMock()
    tool_call_1.function.name = "speak"
    tool_call_1.function.arguments = '{"text": "hello"}'

    tool_call_2 = MagicMock()
    tool_call_2.function.name = "move"
    tool_call_2.function.arguments = '{"direction": "forward"}'

    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content=None, tool_calls=[tool_call_1, tool_call_2])
        )
    ]
    return response


@pytest.fixture(autouse=True)
def mock_avatar_components():
    """Mock all avatar and IO components to prevent Zenoh session creation."""

    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch(
            "llm.plugins.near_ai_llm.AvatarLLMState.trigger_thinking", mock_decorator
        ),
        patch("llm.plugins.near_ai_llm.AvatarLLMState") as mock_avatar_state,
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
    """Fixture providing an initialized NearAILLM instance."""
    return NearAILLM(config, available_actions=None)


class TestNearAILLMInit:
    """Tests for NearAILLM initialization."""

    def test_init_with_config(self, llm, config):
        """Test initialization with provided configuration."""
        assert llm._client.base_url == config.base_url
        assert llm._client.api_key == config.api_key
        assert llm._config.model == config.model

    def test_init_default_base_url(self):
        """Test default base URL when not provided."""
        config = NearAIConfig(api_key="test_key")
        llm = NearAILLM(config, available_actions=None)
        assert "nearai" in str(llm._client.base_url)

    def test_init_default_model(self):
        """Test default model is set when not provided."""
        config = NearAIConfig(api_key="test_key")
        llm = NearAILLM(config, available_actions=None)
        assert llm._config.model is not None
        assert "gpt-oss-120b" in llm._config.model.lower()

    def test_init_requires_api_key(self):
        """Test that initialization fails without API key."""
        config = NearAIConfig(base_url="test_url")
        with pytest.raises(ValueError, match="config file missing api_key"):
            NearAILLM(config, available_actions=None)

    def test_init_with_empty_api_key(self):
        """Test that initialization fails with empty API key."""
        config = NearAIConfig(api_key="")
        with pytest.raises(ValueError, match="config file missing api_key"):
            NearAILLM(config, available_actions=None)

    def test_init_creates_history_manager(self, llm):
        """Test that history manager is initialized."""
        assert llm.history_manager is not None

    def test_init_with_available_actions(self, config):
        """Test initialization with available actions generates function schemas."""
        mock_action = MagicMock()
        mock_action.name = "test_action"
        mock_action.interface = MagicMock()

        with patch("llm.generate_function_schemas_from_actions") as mock_gen:
            mock_gen.return_value = [{"name": "test_action"}]
            llm = NearAILLM(config, available_actions=[mock_action])
            assert len(llm.function_schemas) > 0


class TestNearAILLMAsk:
    """Tests for NearAILLM.ask() method."""

    @pytest.mark.asyncio
    async def test_ask_success_no_tool_calls(self, llm, mock_response):
        """Test successful API request without tool calls returns None."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_response),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_with_tool_calls(self, llm, mock_response_with_tool_calls):
        """Test successful API request with tool calls."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_response_with_tool_calls),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)
            assert result.actions == [Action(type="test_function", value="value1")]

    @pytest.mark.asyncio
    async def test_ask_with_multiple_tool_calls(
        self, llm, mock_response_with_multiple_tool_calls
    ):
        """Test API request with multiple tool calls."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_response_with_multiple_tool_calls),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 2
            assert result.actions[0].type == "speak"
            assert result.actions[1].type == "move"

    @pytest.mark.asyncio
    async def test_ask_api_error(self, llm):
        """Test error handling for API exceptions."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(side_effect=Exception("API error")),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_connection_error(self, llm):
        """Test error handling for connection errors."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(side_effect=ConnectionError("Connection refused")),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_timeout_error(self, llm):
        """Test error handling for timeout errors."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(side_effect=TimeoutError("Request timed out")),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_formats_prompt_correctly(self, llm, mock_response):
        """Test ask() formats the prompt correctly in the request."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")

            # Verify prompt was included in the request
            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert len(formatted_messages) >= 1
            assert formatted_messages[-1]["role"] == "user"
            assert formatted_messages[-1]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_ask_uses_correct_model(self, llm, mock_response):
        """Test ask() uses the configured model."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")

            call_args = mock_parse.call_args
            assert call_args.kwargs.get("model") == llm._config.model

    @pytest.mark.asyncio
    async def test_io_provider_timing(self, llm, mock_response):
        """Test timing metrics collection."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_response),
            )

            await llm.ask("test prompt")
            assert llm.io_provider.llm_start_time is not None
            assert llm.io_provider.llm_end_time is not None
            assert llm.io_provider.llm_end_time >= llm.io_provider.llm_start_time

    @pytest.mark.asyncio
    async def test_ask_sets_llm_prompt(self, llm, mock_response):
        """Test that ask() sets the prompt in io_provider."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_response),
            )

            await llm.ask("my test prompt")
            # io_provider.set_llm_prompt should have been called

    @pytest.mark.asyncio
    async def test_ask_includes_tool_choice(self, llm, mock_response):
        """Test that ask() includes tool_choice parameter."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")

            call_args = mock_parse.call_args
            assert call_args.kwargs.get("tool_choice") == "auto"

    @pytest.mark.asyncio
    async def test_ask_with_function_schemas(
        self, config, mock_response_with_tool_calls
    ):
        """Test ask() includes function schemas when available."""
        llm = NearAILLM(config, available_actions=None)
        llm.function_schemas = [{"type": "function", "function": {"name": "test"}}]

        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response_with_tool_calls)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")

            call_args = mock_parse.call_args
            assert "tools" in call_args.kwargs
