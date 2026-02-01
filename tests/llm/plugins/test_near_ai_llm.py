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
        assert "qwen" in llm._config.model.lower()

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


class TestNearAILLMMessageHandling:
    """Tests for message formatting and handling."""

    @pytest.mark.asyncio
    async def test_ask_with_messages_parameter(self, llm, mock_response):
        """Test ask() with messages parameter."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt", messages=messages)

            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            
            # Should have system message, user message, and the prompt
            assert len(formatted_messages) == 3
            assert formatted_messages[0]["role"] == "system"
            assert formatted_messages[0]["content"] == "You are helpful"
            assert formatted_messages[1]["role"] == "user"
            assert formatted_messages[1]["content"] == "Hello"
            assert formatted_messages[2]["role"] == "user"
            assert formatted_messages[2]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_ask_with_incomplete_messages(self, llm, mock_response):
        """Test ask() with messages missing role or content."""
        messages = [
            {"content": "Hello"},  # missing role
            {"role": "assistant"}  # missing content
        ]
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt", messages=messages)

            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            
            # First message should default to "user" role
            assert formatted_messages[0]["role"] == "user"
            assert formatted_messages[0]["content"] == "Hello"
            
            # Second message should have empty content
            assert formatted_messages[1]["role"] == "assistant"
            assert formatted_messages[1]["content"] == ""

    @pytest.mark.asyncio
    async def test_ask_with_empty_messages_list(self, llm, mock_response):
        """Test ask() with empty messages list."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt", messages=[])

            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            
            # Should only have the prompt message
            assert len(formatted_messages) == 1
            assert formatted_messages[0]["role"] == "user"
            assert formatted_messages[0]["content"] == "test prompt"


class TestNearAILLMEmptyChoices:
    """Tests for empty choices handling."""

    @pytest.mark.asyncio
    async def test_ask_empty_choices(self, llm):
        """Test handling of empty choices in response."""
        mock_resp = MagicMock()
        mock_resp.choices = []
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_resp),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_none_choices(self, llm):
        """Test handling when choices is None."""
        mock_resp = MagicMock()
        mock_resp.choices = None
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=mock_resp),
            )

            result = await llm.ask("test prompt")
            assert result is None


class TestNearAILLMPromptEdgeCases:
    """Tests for prompt edge cases."""

    @pytest.mark.asyncio
    async def test_ask_with_empty_prompt(self, llm, mock_response):
        """Test ask() with empty string prompt."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            result = await llm.ask("")
            
            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert formatted_messages[-1]["content"] == ""

    @pytest.mark.asyncio
    async def test_ask_with_very_long_prompt(self, llm, mock_response):
        """Test ask() with very long prompt."""
        long_prompt = "test " * 10000  # Very long prompt
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask(long_prompt)
            
            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert formatted_messages[-1]["content"] == long_prompt

    @pytest.mark.asyncio
    async def test_ask_with_special_characters(self, llm, mock_response):
        """Test ask() with special characters in prompt."""
        special_prompt = "Test with 特殊字符 émojis 🚀 and\nnewlines\ttabs"
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask(special_prompt)
            
            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert formatted_messages[-1]["content"] == special_prompt

    @pytest.mark.asyncio
    async def test_ask_with_unicode_prompt(self, llm, mock_response):
        """Test ask() with Unicode characters."""
        unicode_prompt = "你好世界 🌍 Привет مرحبا"
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask(unicode_prompt)
            
            call_args = mock_parse.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert formatted_messages[-1]["content"] == unicode_prompt


class TestNearAILLMModelConfiguration:
    """Tests for different model configurations."""

    def test_init_with_qwen_30b_model(self):
        """Test initialization with QWEN_30B model."""
        from llm.plugins.near_ai_llm import NearAIModel
        
        config = NearAIConfig(
            api_key="test_key",
            model=NearAIModel.QWEN_30B_A3B_INSTRUCT_2507
        )
        llm = NearAILLM(config, available_actions=None)
        assert llm._config.model == NearAIModel.QWEN_30B_A3B_INSTRUCT_2507

    def test_init_with_qwen_vl_model(self):
        """Test initialization with QWEN_2_5_VL model."""
        from llm.plugins.near_ai_llm import NearAIModel
        
        config = NearAIConfig(
            api_key="test_key",
            model=NearAIModel.QWEN_2_5_VL_72B_INSTRUCT
        )
        llm = NearAILLM(config, available_actions=None)
        assert llm._config.model == NearAIModel.QWEN_2_5_VL_72B_INSTRUCT

    def test_init_with_qwen_7b_model(self):
        """Test initialization with QWEN_2_5_7B model."""
        from llm.plugins.near_ai_llm import NearAIModel
        
        config = NearAIConfig(
            api_key="test_key",
            model=NearAIModel.QWEN_2_5_7B_INSTRUCT
        )
        llm = NearAILLM(config, available_actions=None)
        assert llm._config.model == NearAIModel.QWEN_2_5_7B_INSTRUCT

    def test_init_with_custom_model_string(self):
        """Test initialization with custom model string."""
        config = NearAIConfig(
            api_key="test_key",
            model="custom-model-v1"
        )
        llm = NearAILLM(config, available_actions=None)
        assert llm._config.model == "custom-model-v1"

    @pytest.mark.asyncio
    async def test_ask_uses_enum_model(self, mock_response):
        """Test that ask() correctly uses enum model value."""
        from llm.plugins.near_ai_llm import NearAIModel
        
        config = NearAIConfig(
            api_key="test_key",
            model=NearAIModel.QWEN_2_5_VL_72B_INSTRUCT
        )
        llm = NearAILLM(config, available_actions=None)
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")
            
            call_args = mock_parse.call_args
            assert call_args.kwargs.get("model") == NearAIModel.QWEN_2_5_VL_72B_INSTRUCT


class TestNearAILLMTimeoutConfiguration:
    """Tests for timeout configuration."""

    @pytest.mark.asyncio
    async def test_ask_timeout_config(self, mock_response):
        """Test that timeout config is passed to API call."""
        config = NearAIConfig(
            api_key="test_key",
            timeout=30
        )
        llm = NearAILLM(config, available_actions=None)
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")
            
            call_args = mock_parse.call_args
            assert call_args.kwargs.get("timeout") == 30

    @pytest.mark.asyncio
    async def test_ask_default_timeout(self, llm, mock_response):
        """Test default timeout behavior."""
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")
            
            call_args = mock_parse.call_args
            # Should have timeout parameter
            assert "timeout" in call_args.kwargs


class TestNearAILLMBaseURLConfiguration:
    """Tests for base URL configuration."""

    def test_init_with_none_base_url(self):
        """Test initialization with None base_url uses default."""
        config = NearAIConfig(api_key="test_key", base_url=None)
        llm = NearAILLM(config, available_actions=None)
        assert "nearai" in str(llm._client.base_url).lower()

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        custom_url = "https://custom.api.example.com/v1"
        config = NearAIConfig(api_key="test_key", base_url=custom_url)
        llm = NearAILLM(config, available_actions=None)
        assert str(llm._client.base_url) == custom_url

    def test_init_with_trailing_slash_base_url(self):
        """Test initialization handles trailing slash in base URL."""
        config = NearAIConfig(
            api_key="test_key",
            base_url="https://api.test.com/"
        )
        llm = NearAILLM(config, available_actions=None)
        assert llm._client.base_url is not None


class TestNearAILLMToolCallConversion:
    """Tests for tool call conversion logic."""

    @pytest.mark.asyncio
    async def test_tool_call_with_invalid_json_arguments(self, llm):
        """Test handling of tool calls with invalid JSON in arguments."""
        tool_call = MagicMock()
        tool_call.function.name = "test_function"
        tool_call.function.arguments = "invalid json {"
        
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[tool_call])
            )
        ]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=response),
            )

            # Should handle gracefully or return None on conversion error
            result = await llm.ask("test prompt")
            # The behavior depends on convert_function_calls_to_actions implementation

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_arguments(self, llm):
        """Test handling of tool calls with empty arguments."""
        tool_call = MagicMock()
        tool_call.function.name = "test_function"
        tool_call.function.arguments = "{}"
        
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[tool_call])
            )
        ]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=response),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)

    @pytest.mark.asyncio
    async def test_tool_call_with_complex_arguments(self, llm):
        """Test handling of tool calls with complex nested arguments."""
        tool_call = MagicMock()
        tool_call.function.name = "complex_function"
        tool_call.function.arguments = '{"nested": {"key": "value"}, "list": [1, 2, 3]}'
        
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[tool_call])
            )
        ]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=response),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)


class TestNearAILLMFunctionSchemas:
    """Tests for function schema handling."""

    @pytest.mark.asyncio
    async def test_ask_without_function_schemas(self, mock_response):
        """Test ask() when no function schemas are available."""
        config = NearAIConfig(api_key="test_key")
        llm = NearAILLM(config, available_actions=None)
        llm.function_schemas = []
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")
            
            call_args = mock_parse.call_args
            tools = call_args.kwargs.get("tools", [])
            assert tools == []

    @pytest.mark.asyncio
    async def test_ask_with_multiple_function_schemas(
        self, config, mock_response_with_tool_calls
    ):
        """Test ask() with multiple function schemas."""
        llm = NearAILLM(config, available_actions=None)
        llm.function_schemas = [
            {"type": "function", "function": {"name": "func1"}},
            {"type": "function", "function": {"name": "func2"}},
            {"type": "function", "function": {"name": "func3"}}
        ]
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response_with_tool_calls)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            await llm.ask("test prompt")
            
            call_args = mock_parse.call_args
            tools = call_args.kwargs.get("tools", [])
            assert len(tools) == 3


class TestNearAILLMLogging:
    """Tests for logging behavior."""

    @pytest.mark.asyncio
    async def test_ask_logs_input(self, llm, mock_response, caplog):
        """Test that ask() logs input prompt and messages."""
        import logging
        
        with caplog.at_level(logging.INFO):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    llm._client.beta.chat.completions,
                    "parse",
                    AsyncMock(return_value=mock_response),
                )

                await llm.ask("test prompt")
                
                assert "NearAI LLM input: test prompt" in caplog.text

    @pytest.mark.asyncio
    async def test_ask_logs_function_calls(
        self, llm, mock_response_with_tool_calls, caplog
    ):
        """Test that ask() logs function calls."""
        import logging
        
        with caplog.at_level(logging.INFO):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    llm._client.beta.chat.completions,
                    "parse",
                    AsyncMock(return_value=mock_response_with_tool_calls),
                )

                await llm.ask("test prompt")
                
                assert "Received" in caplog.text
                assert "function calls" in caplog.text

    @pytest.mark.asyncio
    async def test_ask_logs_error(self, llm, caplog):
        """Test that ask() logs errors."""
        import logging
        
        with caplog.at_level(logging.ERROR):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    llm._client.beta.chat.completions,
                    "parse",
                    AsyncMock(side_effect=Exception("Test error")),
                )

                await llm.ask("test prompt")
                
                assert "NearAI API error" in caplog.text
                assert "Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_choices_logs_warning(self, llm, caplog):
        """Test that empty choices logs a warning."""
        import logging
        
        mock_resp = MagicMock()
        mock_resp.choices = []
        
        with caplog.at_level(logging.WARNING):
            with pytest.MonkeyPatch.context() as m:
                m.setattr(
                    llm._client.beta.chat.completions,
                    "parse",
                    AsyncMock(return_value=mock_resp),
                )

                await llm.ask("test prompt")
                
                assert "NearAI API returned empty choices" in caplog.text


class TestNearAILLMResponseMessage:
    """Tests for different response message scenarios."""

    @pytest.mark.asyncio
    async def test_ask_with_message_content_and_no_tool_calls(self, llm):
        """Test response with content but no tool calls."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Just a text response",
                    tool_calls=None
                )
            )
        ]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=response),
            )

            result = await llm.ask("test prompt")
            # Should return None as there are no tool calls
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_with_none_message_content(self, llm):
        """Test response with None content and no tool calls."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=None
                )
            )
        ]
        
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.beta.chat.completions,
                "parse",
                AsyncMock(return_value=response),
            )

            result = await llm.ask("test prompt")
            assert result is None


class TestNearAILLMConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_asks(self, llm, mock_response):
        """Test multiple concurrent ask() calls."""
        import asyncio
        
        with pytest.MonkeyPatch.context() as m:
            mock_parse = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.beta.chat.completions, "parse", mock_parse)

            # Make multiple concurrent calls
            tasks = [
                llm.ask(f"prompt {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should return None (no tool calls in mock_response)
            assert all(r is None for r in results)
            
            # Should have been called 5 times
            assert mock_parse.call_count == 5