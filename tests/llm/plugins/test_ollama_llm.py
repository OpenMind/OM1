from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from llm.plugins.ollama_llm import OllamaLLM, OllamaLLMConfig


@pytest.fixture
def config():
    return OllamaLLMConfig(
        base_url="http://localhost:11434",
        model="llama3.2",
        temperature=0.7,
        timeout=60,
    )


@pytest.fixture
def mock_response():
    """Fixture providing a valid mock Ollama API response"""
    return {
        "message": {
            "role": "assistant",
            "content": "Hello!",
            "tool_calls": None,
        }
    }


@pytest.fixture
def mock_response_with_tool_calls():
    """Fixture providing a mock API response with tool calls"""
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "test_function",
                        "arguments": {"arg1": "value1"},
                    }
                }
            ],
        }
    }


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
        patch("llm.plugins.ollama_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.ollama_llm.AvatarLLMState") as mock_avatar_state,
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
    return OllamaLLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    """Test initialization with provided configuration"""
    assert llm._base_url == config.base_url.rstrip("/")
    assert llm._config.model == config.model
    assert llm._config.temperature == config.temperature


@pytest.mark.asyncio
async def test_init_default_config():
    """Test initialization with default configuration"""
    config = OllamaLLMConfig()
    llm = OllamaLLM(config, available_actions=None)
    assert llm._base_url == "http://localhost:11434"
    assert llm._config.model == "llama3.2"


@pytest.mark.asyncio
async def test_ask_success(llm, mock_response):
    """Test successful API request and response parsing"""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_response

    with patch.object(llm._client, "post", AsyncMock(return_value=mock_http_response)):
        result = await llm.ask("test prompt")
        assert result is None  # No tool calls, content not JSON


@pytest.mark.asyncio
async def test_ask_with_tool_calls(llm, mock_response_with_tool_calls):
    """Test successful API request with tool calls"""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_response_with_tool_calls

    with patch.object(llm._client, "post", AsyncMock(return_value=mock_http_response)):
        result = await llm.ask("test prompt")
        assert isinstance(result, CortexOutputModel)
        assert result.actions == [Action(type="test_function", value="value1")]


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    """Test handling of API errors"""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 500
    mock_http_response.text = "Internal Server Error"

    with patch.object(llm._client, "post", AsyncMock(return_value=mock_http_response)):
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_connection_error(llm):
    """Test handling of connection errors"""
    import httpx

    with patch.object(
        llm._client,
        "post",
        AsyncMock(side_effect=httpx.ConnectError("Connection refused")),
    ):
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_timeout_error(llm):
    """Test handling of timeout errors"""
    import httpx

    with patch.object(
        llm._client, "post", AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    ):
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_convert_tools_to_ollama_format(llm):
    """Test tool schema conversion with no actions"""
    tools = llm._convert_tools_to_ollama_format()
    assert isinstance(tools, list)
    assert len(tools) == 0  # No actions configured


@pytest.mark.asyncio
async def test_close(llm):
    """Test HTTP client cleanup"""
    with patch.object(llm._client, "aclose", AsyncMock()) as mock_close:
        await llm.close()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_convert_tools_with_schemas(config):
    """Test tool conversion with actual function schemas"""
    schemas = [
        {
            "function": {
                "name": "test_func",
                "description": "A test function",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    ]
    llm = OllamaLLM(config, available_actions=None)
    llm.function_schemas = schemas
    tools = llm._convert_tools_to_ollama_format()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "test_func"


@pytest.mark.asyncio
async def test_ask_json_parse_error(llm):
    """Test handling of JSON parsing errors"""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.side_effect = ValueError("Invalid JSON")

    with patch.object(llm._client, "post", AsyncMock(return_value=mock_http_response)):
        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_base_url_trailing_slash():
    """Test base_url with trailing slash is handled correctly"""
    config = OllamaLLMConfig(base_url="http://localhost:11434/")
    llm = OllamaLLM(config, available_actions=None)
    assert llm._base_url == "http://localhost:11434"
    assert llm._chat_url == "http://localhost:11434/api/chat"


@pytest.mark.asyncio
async def test_tool_calls_with_string_arguments(llm):
    """Test tool calls where arguments is a JSON string"""
    mock_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "test_function",
                        "arguments": '{"arg1": "value1"}',  # 字串格式
                    }
                }
            ],
        }
    }
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_response

    with patch.object(llm._client, "post", AsyncMock(return_value=mock_http_response)):
        result = await llm.ask("test prompt")
        assert isinstance(result, CortexOutputModel)
