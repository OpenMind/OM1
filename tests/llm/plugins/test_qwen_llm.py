from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.qwen_llm import QwenLLM, _parse_qwen_tool_calls


class DummyOutputModel(BaseModel):
    test_field: str


class TestParseQwenToolCalls:
    """Tests for the _parse_qwen_tool_calls helper function."""

    def test_parse_single_tool_call(self):
        """Test parsing a single valid tool call block."""
        text = (
            '<tool_call>{"name": "speak", "arguments": {"text": "hello"}}</tool_call>'
        )
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "speak"
        assert result[0]["function"]["arguments"] == '{"text": "hello"}'

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool call blocks."""
        text = """
        <tool_call>{"name": "speak", "arguments": {"text": "hello"}}</tool_call>
        <tool_call>{"name": "move", "arguments": {"direction": "forward"}}</tool_call>
        """
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "speak"
        assert result[1]["function"]["name"] == "move"

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        assert _parse_qwen_tool_calls("") == []

    def test_parse_no_tool_calls(self):
        """Test parsing text without tool calls returns empty list."""
        text = "This is just regular text without any tool calls."
        assert _parse_qwen_tool_calls(text) == []

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON inside tool_call tags is skipped."""
        text = "<tool_call>this is not valid json</tool_call>"
        result = _parse_qwen_tool_calls(text)
        assert result == []

    def test_parse_missing_name_field(self):
        """Test parsing tool call without 'name' field is skipped."""
        text = '<tool_call>{"arguments": {"text": "hello"}}</tool_call>'
        result = _parse_qwen_tool_calls(text)
        assert result == []

    def test_parse_empty_arguments(self):
        """Test parsing tool call with empty arguments."""
        text = '<tool_call>{"name": "stop", "arguments": {}}</tool_call>'
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "stop"
        assert result[0]["function"]["arguments"] == "{}"

    def test_parse_missing_arguments(self):
        """Test parsing tool call without arguments field defaults to empty dict."""
        text = '<tool_call>{"name": "stop"}</tool_call>'
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "stop"
        assert result[0]["function"]["arguments"] == "{}"

    def test_parse_with_whitespace(self):
        """Test parsing tool calls with extra whitespace."""
        text = """
        <tool_call>
            {"name": "speak", "arguments": {"text": "hello"}}
        </tool_call>
        """
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "speak"

    def test_parse_non_string_input(self):
        """Test parsing non-string input returns empty list."""
        assert _parse_qwen_tool_calls(None) == []  # type: ignore
        assert _parse_qwen_tool_calls(123) == []  # type: ignore
        assert _parse_qwen_tool_calls([]) == []  # type: ignore

    def test_parse_unicode_arguments(self):
        """Test parsing tool calls with unicode characters."""
        text = '<tool_call>{"name": "speak", "arguments": {"text": "Merhaba Dünya"}}</tool_call>'
        result = _parse_qwen_tool_calls(text)

        assert len(result) == 1
        assert "Merhaba" in result[0]["function"]["arguments"]

    def test_unique_call_ids(self):
        """Test that each parsed tool call has a unique ID."""
        text = """
        <tool_call>{"name": "a", "arguments": {}}</tool_call>
        <tool_call>{"name": "b", "arguments": {}}</tool_call>
        <tool_call>{"name": "c", "arguments": {}}</tool_call>
        """
        result = _parse_qwen_tool_calls(text)

        ids = [tc["id"] for tc in result]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"


@pytest.fixture
def config():
    """Fixture providing a basic LLM configuration."""
    return LLMConfig(model="test-qwen-model")


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
    """Fixture providing a mock API response with native tool calls."""
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
def mock_response_with_xml_tool_calls():
    """Fixture providing a mock response with Qwen-style XML tool calls."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='<tool_call>{"name": "test_function", "arguments": {"arg1": "value1"}}</tool_call>',
                tool_calls=None,
            )
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
        patch("llm.plugins.qwen_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.qwen_llm.AvatarLLMState") as mock_avatar_state,
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
    """Fixture providing an initialized QwenLLM instance."""
    return QwenLLM(config, available_actions=None)


class TestQwenLLMInit:
    """Tests for QwenLLM initialization."""

    def test_init_with_config(self, llm, config):
        """Test initialization with provided configuration."""
        assert llm._config.model == config.model
        assert llm._client.base_url == "http://127.0.0.1:8000/v1/"

    def test_init_default_model(self):
        """Test default model is set when not provided."""
        config = LLMConfig()
        llm = QwenLLM(config, available_actions=None)
        assert llm._config.model is not None
        assert "Qwen" in llm._config.model

    def test_init_extra_body_config(self, llm):
        """Test extra_body is configured to disable thinking mode."""
        assert llm._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_init_placeholder_api_key(self, llm):
        """Test that a placeholder API key is used for local server."""
        assert llm._client.api_key == "placeholder_key"

    def test_init_with_available_actions(self, config):
        """Test initialization with available actions generates function schemas."""
        mock_action = MagicMock()
        mock_action.name = "test_action"
        mock_action.interface = MagicMock()

        with patch("llm.generate_function_schemas_from_actions") as mock_gen:
            mock_gen.return_value = [{"name": "test_action"}]
            llm = QwenLLM(config, available_actions=[mock_action])
            assert len(llm.function_schemas) > 0


class TestQwenLLMAsk:
    """Tests for QwenLLM.ask() method."""

    @pytest.mark.asyncio
    async def test_ask_success_no_tool_calls(self, llm, mock_response):
        """Test successful API request without tool calls returns None."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=mock_response),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_with_native_tool_calls(self, llm, mock_response_with_tool_calls):
        """Test successful API request with native tool calls."""
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
    async def test_ask_with_xml_tool_calls(
        self, llm, mock_response_with_xml_tool_calls
    ):
        """Test fallback parsing of Qwen-style XML tool calls."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=mock_response_with_xml_tool_calls),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 1
            assert result.actions[0].type == "test_function"

    @pytest.mark.asyncio
    async def test_ask_api_error(self, llm):
        """Test error handling for API exceptions."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(side_effect=Exception("Connection refused")),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_formats_prompt_correctly(self, llm, mock_response):
        """Test ask() formats the prompt correctly in the request."""
        with pytest.MonkeyPatch.context() as m:
            mock_create = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.chat.completions, "create", mock_create)

            await llm.ask("test prompt")

            # Verify prompt was included in the request
            call_args = mock_create.call_args
            formatted_messages = call_args.kwargs.get("messages", [])
            assert len(formatted_messages) >= 1
            assert formatted_messages[-1]["role"] == "user"
            assert formatted_messages[-1]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_io_provider_timing(self, llm, mock_response):
        """Test timing metrics collection."""
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
    async def test_ask_includes_extra_body(self, llm, mock_response):
        """Test that extra_body is included in API request."""
        with pytest.MonkeyPatch.context() as m:
            mock_create = AsyncMock(return_value=mock_response)
            m.setattr(llm._client.chat.completions, "create", mock_create)

            await llm.ask("test prompt")

            call_args = mock_create.call_args
            assert call_args.kwargs.get("extra_body") == {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    @pytest.mark.asyncio
    async def test_ask_with_function_schemas(
        self, config, mock_response_with_tool_calls
    ):
        """Test ask() includes function schemas when available."""
        # Create LLM with mock function schemas
        llm = QwenLLM(config, available_actions=None)
        llm.function_schemas = [{"type": "function", "function": {"name": "test"}}]

        with pytest.MonkeyPatch.context() as m:
            mock_create = AsyncMock(return_value=mock_response_with_tool_calls)
            m.setattr(llm._client.chat.completions, "create", mock_create)

            await llm.ask("test prompt")

            call_args = mock_create.call_args
            assert "tools" in call_args.kwargs
            assert call_args.kwargs.get("tool_choice") == "required"
