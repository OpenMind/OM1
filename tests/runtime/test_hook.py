import asyncio
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from runtime.hook import (
    ActionHookConfig,
    ActionHookHandler,
    CommandHookConfig,
    CommandHookHandler,
    FunctionHookConfig,
    FunctionHookHandler,
    LifecycleHook,
    LifecycleHookHandler,
    LifecycleHookType,
    MessageHookConfig,
    MessageHookHandler,
    create_hook_handler,
    execute_lifecycle_hooks,
    parse_lifecycle_hooks,
)


@pytest.fixture
def sample_message_hook():
    """Sample message hook configuration."""
    return LifecycleHook(
        hook_type=LifecycleHookType.ON_ENTRY,
        handler_type="message",
        handler_config={
            "message": "Entering mode: {mode_name}",
        },
        async_execution=True,
        timeout_seconds=5.0,
        on_failure="ignore",
        priority=1,
    )


@pytest.fixture
def sample_command_hook():
    """Sample command hook configuration."""
    return LifecycleHook(
        hook_type=LifecycleHookType.ON_EXIT,
        handler_type="command",
        handler_config={"command": "echo 'Exiting mode: {mode_name}'"},
        async_execution=True,
        timeout_seconds=10.0,
        on_failure="abort",
        priority=2,
    )


@pytest.fixture
def sample_function_hook():
    """Sample function hook configuration."""
    return LifecycleHook(
        hook_type=LifecycleHookType.ON_STARTUP,
        handler_type="function",
        handler_config={
            "function": "startup_handler",
            "module_name": "test_module",
        },
        async_execution=True,
        timeout_seconds=15.0,
        on_failure="ignore",
        priority=0,
    )


@pytest.fixture
def sample_action_hook():
    """Sample action hook configuration."""
    return LifecycleHook(
        hook_type=LifecycleHookType.ON_SHUTDOWN,
        handler_type="action",
        handler_config={
            "action_type": "test_action",
            "action_config": {"param": "value"},
        },
        async_execution=False,
        timeout_seconds=30.0,
        on_failure="abort",
        priority=5,
    )


@pytest.fixture
def sample_context():
    """Sample context for hook execution."""
    return {
        "mode_name": "test_mode",
        "user_id": "user123",
        "timestamp": "2025-01-01T00:00:00Z",
    }


def test_hook_type_values():
    """Test that all hook type values are correctly defined."""
    assert LifecycleHookType.ON_ENTRY.value == "on_entry"
    assert LifecycleHookType.ON_EXIT.value == "on_exit"
    assert LifecycleHookType.ON_STARTUP.value == "on_startup"
    assert LifecycleHookType.ON_SHUTDOWN.value == "on_shutdown"
    assert LifecycleHookType.ON_TIMEOUT.value == "on_timeout"


def test_basic_hook_creation():
    """Test basic hook creation with required fields."""
    hook = LifecycleHook(
        hook_type=LifecycleHookType.ON_ENTRY,
        handler_type="message",
        handler_config={"message": "test"},
    )

    assert hook.hook_type == LifecycleHookType.ON_ENTRY
    assert hook.handler_type == "message"
    assert hook.handler_config == {"message": "test"}
    assert hook.async_execution is True
    assert hook.timeout_seconds == 5.0
    assert hook.on_failure == "ignore"
    assert hook.priority == 0


def test_hook_with_custom_values(sample_command_hook):
    """Test hook creation with custom values."""
    assert sample_command_hook.hook_type == LifecycleHookType.ON_EXIT
    assert sample_command_hook.handler_type == "command"
    assert sample_command_hook.timeout_seconds == 10.0
    assert sample_command_hook.on_failure == "abort"
    assert sample_command_hook.priority == 2


def test_base_handler_creation():
    """Test base handler creation."""
    config = MessageHookConfig(message="test value")
    handler = LifecycleHookHandler(config)
    assert isinstance(handler.config, MessageHookConfig)


@pytest.mark.asyncio
async def test_base_handler_execute_not_implemented():
    """Test that base handler execute method raises NotImplementedError."""
    handler = LifecycleHookHandler(MessageHookConfig())
    with pytest.raises(NotImplementedError):
        await handler.execute({})


def test_message_handler_creation():
    """Test message handler creation."""
    config = MessageHookConfig(message="test message")
    handler = MessageHookHandler(config)
    assert isinstance(handler.config, MessageHookConfig)
    assert handler.config.message == "test message"


@pytest.mark.asyncio
async def test_message_handler_basic_execution(sample_context):
    """Test basic message handler execution."""
    config = MessageHookConfig(message="Mode: {mode_name}")
    handler = MessageHookHandler(config)

    mock_tts = Mock()
    mock_tts.add_pending_message = Mock()

    with patch("runtime.hook.logging") as mock_logging:
        with patch("runtime.hook.ElevenLabsTTSProvider", return_value=mock_tts):
            result = await handler.execute(sample_context)
            assert result is True
            mock_logging.info.assert_called_once_with(
                "Lifecycle hook message: Mode: test_mode"
            )
            mock_tts.add_pending_message.assert_called_once_with("Mode: test_mode")


@pytest.mark.asyncio
async def test_message_handler_with_announcement(sample_context):
    """Test message handler with TTS announcement."""
    config = MessageHookConfig(message="Mode: {mode_name}")
    handler = MessageHookHandler(config)

    mock_tts = Mock()
    mock_tts.add_pending_message = Mock()

    with patch("runtime.hook.logging") as mock_logging:
        with patch("runtime.hook.ElevenLabsTTSProvider", return_value=mock_tts):
            result = await handler.execute(sample_context)
            assert result is True
            mock_logging.info.assert_called_once()
            mock_tts.add_pending_message.assert_called_once_with("Mode: test_mode")


@pytest.mark.asyncio
async def test_message_handler_tts_import_error(sample_context):
    """Test message handler when TTS provider is not available."""
    config = MessageHookConfig(message="Mode: {mode_name}")
    handler = MessageHookHandler(config)

    with patch("runtime.hook.logging") as mock_logging:
        with patch("runtime.hook.ElevenLabsTTSProvider", side_effect=ImportError):
            result = await handler.execute(sample_context)
            assert result is False
            mock_logging.error.assert_called_once_with("Error adding TTS message: ")


@pytest.mark.asyncio
async def test_message_handler_format_error():
    """Test message handler with format error."""
    config = MessageHookConfig(message="Invalid format: {nonexistent_key}")
    handler = MessageHookHandler(config)
    context = {"mode_name": "test"}

    with patch("runtime.hook.logging") as mock_logging:
        result = await handler.execute(context)
        assert result is False
        mock_logging.error.assert_called_once()


@pytest.mark.asyncio
async def test_message_handler_no_message():
    """Test message handler with no message configured."""
    config = MessageHookConfig()
    handler = MessageHookHandler(config)

    result = await handler.execute({})
    assert result is True


@pytest.mark.asyncio
async def test_message_handler_empty_message():
    """Test message handler with empty message."""
    config = MessageHookConfig(message="")
    handler = MessageHookHandler(config)

    result = await handler.execute({})
    assert result is True


def test_message_handler_default_tts_provider():
    """Test that default TTS provider is elevenlabs."""
    config = MessageHookConfig(message="test")
    handler = MessageHookHandler(config)
    assert handler.config.tts_provider == "elevenlabs"


def test_message_handler_create_elevenlabs_provider():
    """Test creating ElevenLabs TTS provider with default settings."""
    config = MessageHookConfig(
        message="test",
        tts_provider="elevenlabs",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.ElevenLabsTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=None,
            elevenlabs_api_key=None,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000",
            enable_tts_interrupt=False,
        )


def test_message_handler_create_elevenlabs_provider_with_custom_config():
    """Test creating ElevenLabs TTS provider with custom configuration."""
    config = MessageHookConfig(
        message="test",
        tts_provider="elevenlabs",
        url="https://custom.url/tts",
        api_key="test_api_key",
        elevenlabs_api_key="test_elevenlabs_key",
        voice_id="custom_voice",
        model_id="custom_model",
        output_format="wav_44100",
        enable_tts_interrupt=True,
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.ElevenLabsTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="https://custom.url/tts",
            api_key="test_api_key",
            elevenlabs_api_key="test_elevenlabs_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="wav_44100",
            enable_tts_interrupt=True,
        )


def test_message_handler_create_kokoro_provider():
    """Test creating Kokoro TTS provider with default settings."""
    config = MessageHookConfig(
        message="test",
        tts_provider="kokoro",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.KokoroTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key=None,
            voice_id="af_bella",
            model_id="kokoro",
            output_format="pcm",
            rate=24000,
            enable_tts_interrupt=False,
        )


def test_message_handler_create_kokoro_provider_with_custom_config():
    """Test creating Kokoro TTS provider with custom configuration."""
    config = MessageHookConfig(
        message="test",
        tts_provider="kokoro",
        url="http://custom.host:8880/v1",
        api_key="test_api_key",
        voice_id="af_sky",
        model_id="kokoro_v2",
        output_format="wav",
        rate=48000,
        enable_tts_interrupt=True,
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.KokoroTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="http://custom.host:8880/v1",
            api_key="test_api_key",
            voice_id="af_sky",
            model_id="kokoro_v2",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
        )


def test_message_handler_create_riva_provider():
    """Test creating Riva TTS provider with default settings."""
    config = MessageHookConfig(
        message="test",
        tts_provider="riva",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.RivaTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="http://127.0.0.1:50051",
            api_key=None,
        )


def test_message_handler_create_riva_provider_with_custom_config():
    """Test creating Riva TTS provider with custom configuration."""
    config = MessageHookConfig(
        message="test",
        tts_provider="riva",
        url="http://custom.host:50051",
        api_key="test_api_key",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.RivaTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once_with(
            url="http://custom.host:50051",
            api_key="test_api_key",
        )


def test_message_handler_unsupported_provider():
    """Test creating TTS provider with unsupported provider type."""
    config = MessageHookConfig(
        message="test",
        tts_provider="unsupported_provider",
    )
    handler = MessageHookHandler(config)

    with pytest.raises(
        ValueError, match="Unsupported TTS provider: unsupported_provider"
    ):
        handler._create_tts_provider()


def test_message_handler_case_insensitive_provider():
    """Test that provider type is case-insensitive."""
    config = MessageHookConfig(
        message="test",
        tts_provider="ELEVENLABS",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.ElevenLabsTTSProvider") as mock_provider:
        handler._create_tts_provider()
        mock_provider.assert_called_once()


@pytest.mark.asyncio
async def test_message_handler_execute_with_kokoro(sample_context):
    """Test message handler execution with Kokoro provider."""
    config = MessageHookConfig(
        message="Mode: {mode_name}",
        tts_provider="kokoro",
    )
    handler = MessageHookHandler(config)

    mock_tts = Mock()
    mock_tts.start = Mock()
    mock_tts.add_pending_message = Mock()

    with patch("runtime.hook.logging") as mock_logging:
        with patch("runtime.hook.KokoroTTSProvider", return_value=mock_tts):
            result = await handler.execute(sample_context)
            assert result is True
            mock_logging.info.assert_called_once_with(
                "Lifecycle hook message: Mode: test_mode"
            )
            mock_tts.start.assert_called_once()
            mock_tts.add_pending_message.assert_called_once_with("Mode: test_mode")


@pytest.mark.asyncio
async def test_message_handler_execute_with_riva(sample_context):
    """Test message handler execution with Riva provider."""
    config = MessageHookConfig(
        message="Mode: {mode_name}",
        tts_provider="riva",
    )
    handler = MessageHookHandler(config)

    mock_tts = Mock()
    mock_tts.start = Mock()
    mock_tts.add_pending_message = Mock()

    with patch("runtime.hook.logging") as mock_logging:
        with patch("runtime.hook.RivaTTSProvider", return_value=mock_tts):
            result = await handler.execute(sample_context)
            assert result is True
            mock_logging.info.assert_called_once_with(
                "Lifecycle hook message: Mode: test_mode"
            )
            mock_tts.start.assert_called_once()
            mock_tts.add_pending_message.assert_called_once_with("Mode: test_mode")


@pytest.mark.asyncio
async def test_message_handler_execute_with_unsupported_provider(sample_context):
    """Test message handler execution with unsupported provider fails gracefully."""
    config = MessageHookConfig(
        message="Mode: {mode_name}",
        tts_provider="invalid",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.logging") as mock_logging:
        result = await handler.execute(sample_context)
        assert result is False
        mock_logging.error.assert_called_once()
        error_call = mock_logging.error.call_args[0][0]
        assert "Error adding TTS message:" in error_call


@pytest.mark.asyncio
async def test_message_handler_execute_provider_exception(sample_context):
    """Test message handler execution when provider initialization fails."""
    config = MessageHookConfig(
        message="Mode: {mode_name}",
        tts_provider="elevenlabs",
    )
    handler = MessageHookHandler(config)

    with patch("runtime.hook.logging") as mock_logging:
        with patch(
            "runtime.hook.ElevenLabsTTSProvider",
            side_effect=Exception("Provider init failed"),
        ):
            result = await handler.execute(sample_context)
            assert result is False
            mock_logging.error.assert_called_once()
            error_call = mock_logging.error.call_args[0][0]
            assert "Error adding TTS message:" in error_call


def test_command_handler_creation():
    """Test command handler creation."""
    config = CommandHookConfig(command="echo test")
    handler = CommandHookHandler(config)
    assert isinstance(handler.config, CommandHookConfig)
    assert handler.config.command == "echo test"


@pytest.mark.asyncio
async def test_command_handler_successful_execution(sample_context):
    """Test successful command execution."""
    config = CommandHookConfig(command="echo 'Mode: {mode_name}'")
    handler = CommandHookHandler(config)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"Mode: test_mode\n", b"")
    mock_process.returncode = 0

    with patch(
        "runtime.hook.asyncio.create_subprocess_shell",
        return_value=mock_process,
    ):
        with patch("runtime.hook.logging") as mock_logging:
            result = await handler.execute(sample_context)
            assert result is True
            mock_logging.info.assert_called_once_with(
                "Hook command output: Mode: test_mode"
            )


@pytest.mark.asyncio
async def test_command_handler_failed_execution(sample_context):
    """Test failed command execution."""
    config = CommandHookConfig(command="false")  # Command that always fails
    handler = CommandHookHandler(config)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Command failed")
    mock_process.returncode = 1

    with patch(
        "runtime.hook.asyncio.create_subprocess_shell",
        return_value=mock_process,
    ):
        with patch("runtime.hook.logging") as mock_logging:
            result = await handler.execute(sample_context)
            assert result is False
            mock_logging.error.assert_called_once()


@pytest.mark.asyncio
async def test_command_handler_no_command():
    """Test command handler with no command specified."""
    config = CommandHookConfig()
    handler = CommandHookHandler(config)

    with patch("runtime.hook.logging") as mock_logging:
        result = await handler.execute({})
        assert result is False
        mock_logging.warning.assert_called_once_with(
            "No command specified for command hook"
        )


@pytest.mark.asyncio
async def test_command_handler_empty_command():
    """Test command handler with empty command."""
    config = CommandHookConfig(command="")
    handler = CommandHookHandler(config)

    with patch("runtime.hook.logging") as mock_logging:
        result = await handler.execute({})
        assert result is False
        mock_logging.warning.assert_called_once()


@pytest.mark.asyncio
async def test_command_handler_execution_exception(sample_context):
    """Test command handler with execution exception."""
    config = CommandHookConfig(command="echo test")
    handler = CommandHookHandler(config)

    with patch(
        "runtime.hook.asyncio.create_subprocess_shell",
        side_effect=OSError("Permission denied"),
    ):
        with patch("runtime.hook.logging") as mock_logging:
            result = await handler.execute(sample_context)
            assert result is False
            mock_logging.error.assert_called_once()


@pytest.mark.asyncio
async def test_command_handler_successful_no_output(sample_context):
    """Test successful command with no output."""
    config = CommandHookConfig(command="true")  # Command that succeeds with no output
    handler = CommandHookHandler(config)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch(
        "runtime.hook.asyncio.create_subprocess_shell",
        return_value=mock_process,
    ):
        result = await handler.execute(sample_context)
        assert result is True


def test_function_handler_creation():
    """Test function handler creation."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)
    assert isinstance(handler.config, FunctionHookConfig)
    assert handler.config.function == "test_func"
    assert handler.config.module_name == "test_module"


@pytest.mark.asyncio
async def test_function_handler_no_function():
    """Test function handler with no function specified."""
    with pytest.raises(Exception):
        config = FunctionHookConfig(module_name="test_module")  # type: ignore
        FunctionHookHandler(config)


@pytest.mark.asyncio
async def test_function_handler_no_module():
    """Test function handler with no module specified."""
    with pytest.raises(Exception):
        config = FunctionHookConfig(function="test_func")  # type: ignore
        FunctionHookHandler(config)


@pytest.mark.asyncio
async def test_function_handler_successful_sync_execution(sample_context):
    """Test successful synchronous function execution."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    def mock_function(context):
        return True

    with patch.object(handler, "_find_function_in_module", return_value=mock_function):
        result = await handler.execute(sample_context)
        assert result is True


@pytest.mark.asyncio
async def test_function_handler_successful_async_execution(sample_context):
    """Test successful asynchronous function execution."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    async def mock_async_function(context):
        return True

    with patch.object(
        handler, "_find_function_in_module", return_value=mock_async_function
    ):
        result = await handler.execute(sample_context)
        assert result is True


@pytest.mark.asyncio
async def test_function_handler_function_returns_false(sample_context):
    """Test function that returns False."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    def mock_function(context):
        return False

    with patch.object(handler, "_find_function_in_module", return_value=mock_function):
        result = await handler.execute(sample_context)
        assert result is False


@pytest.mark.asyncio
async def test_function_handler_function_returns_none(sample_context):
    """Test function that returns None (should be treated as success)."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    def mock_function(context):
        return None

    with patch.object(handler, "_find_function_in_module", return_value=mock_function):
        result = await handler.execute(sample_context)
        assert result is True


@pytest.mark.asyncio
async def test_function_handler_function_not_found(sample_context):
    """Test function handler when function is not found."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    with patch.object(handler, "_find_function_in_module", return_value=None):
        result = await handler.execute(sample_context)
        assert result is False


@pytest.mark.asyncio
async def test_function_handler_execution_exception(sample_context):
    """Test function handler with execution exception."""
    config = FunctionHookConfig(function="test_func", module_name="test_module")
    handler = FunctionHookHandler(config)

    def mock_function(context):
        raise ValueError("Test error")

    with patch.object(handler, "_find_function_in_module", return_value=mock_function):
        with patch("runtime.hook.logging") as mock_logging:
            result = await handler.execute(sample_context)
            assert result is False
            mock_logging.error.assert_called_once()


def test_find_function_in_module_hooks_dir_not_found():
    """Test function search when hooks directory doesn't exist."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    with patch("runtime.hook.os.path.exists", return_value=False):
        with patch("runtime.hook.logging") as mock_logging:
            result = handler._find_function_in_module("test_module", "test_func")
            assert result is None
            mock_logging.error.assert_called_once()


def test_find_function_in_module_file_not_found():
    """Test function search when module file doesn't exist."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    with patch("runtime.hook.os.path.exists", side_effect=[True, False]):
        with patch("runtime.hook.logging") as mock_logging:
            result = handler._find_function_in_module("test_module", "test_func")
            assert result is None
            mock_logging.error.assert_called_once()


def test_find_function_in_module_function_not_in_file():
    """Test function search when function is not found in file."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    file_content = "def other_function():\n    pass"

    with patch("runtime.hook.os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch("runtime.hook.logging") as mock_logging:
                result = handler._find_function_in_module("test_module", "test_func")
                assert result is None
                mock_logging.error.assert_called_once()


def test_find_function_in_module_import_error():
    """Test function search with import error."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    file_content = "def test_func():\n    pass"

    with patch("runtime.hook.os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch(
                "runtime.hook.importlib.import_module",
                side_effect=ImportError("Module not found"),
            ):
                with patch("runtime.hook.logging") as mock_logging:
                    result = handler._find_function_in_module(
                        "test_module", "test_func"
                    )
                    assert result is None
                    mock_logging.error.assert_called_once()


def test_find_function_in_module_successful():
    """Test successful function search and import."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    file_content = "def test_func():\n    pass"

    def mock_function():
        pass

    mock_module = Mock()
    mock_module.test_func = mock_function

    with patch("runtime.hook.os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch(
                "runtime.hook.importlib.import_module",
                return_value=mock_module,
            ):
                with patch("runtime.hook.hasattr", return_value=True):
                    result = handler._find_function_in_module(
                        "test_module", "test_func"
                    )
                    assert result == mock_function


def test_find_function_in_module_async_function():
    """Test finding async function."""
    handler = FunctionHookHandler(
        FunctionHookConfig(function="test_func", module_name="test_module")
    )

    file_content = "async def test_func():\n    pass"

    async def mock_async_function():
        pass

    mock_module = Mock()
    mock_module.test_func = mock_async_function

    with patch("runtime.hook.os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=file_content)):
            with patch(
                "runtime.hook.importlib.import_module",
                return_value=mock_module,
            ):
                with patch("runtime.hook.hasattr", return_value=True):
                    result = handler._find_function_in_module(
                        "test_module", "test_func"
                    )
                    assert result == mock_async_function


class TestActionHookHandler:
    """Test cases for ActionHookHandler."""

    def test_action_handler_creation(self):
        """Test action handler creation."""
        config = ActionHookConfig(action_type="test_action", action_config={})
        handler = ActionHookHandler(config)
        assert isinstance(handler.config, ActionHookConfig)
        assert handler.config.action_type == "test_action"
        assert handler.action is None

    @pytest.mark.asyncio
    async def test_action_handler_no_action_type(self):
        """Test action handler with no action type specified."""
        with pytest.raises(Exception):
            config = ActionHookConfig(action_config={})  # type: ignore
            ActionHookHandler(config)

    @pytest.mark.asyncio
    async def test_action_handler_action_load_error(self, sample_context):
        """Test action handler with action loading error."""
        config = ActionHookConfig(action_type="nonexistent_action", action_config={})
        handler = ActionHookHandler(config)

        with patch("actions.load_action", side_effect=ImportError("Action not found")):
            with patch("runtime.hook.logging") as mock_logging:
                result = await handler.execute(sample_context)
                assert result is False
                mock_logging.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_handler_successful_execution(self, sample_context):
        """Test successful action execution."""
        config = ActionHookConfig(
            action_type="test_action", action_config={"param": "value"}
        )
        handler = ActionHookHandler(config)

        # Mock action and connector
        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(return_value=None)
        mock_action = Mock()
        mock_action.connector = mock_connector

        with patch("actions.load_action", return_value=mock_action):
            result = await handler.execute(sample_context)
            assert result is True
            mock_connector.connect.assert_called_once_with(
                sample_context.get("input_data")
            )

    @pytest.mark.asyncio
    async def test_action_handler_execution_error(self, sample_context):
        """Test action handler with execution error."""
        config = ActionHookConfig(action_type="test_action", action_config={})
        handler = ActionHookHandler(config)

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(side_effect=Exception("Connection failed"))
        mock_action = Mock()
        mock_action.connector = mock_connector

        with patch("actions.load_action", return_value=mock_action):
            with patch("runtime.hook.logging") as mock_logging:
                result = await handler.execute(sample_context)
                assert result is False
                mock_logging.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_handler_reuse_action(self, sample_context):
        """Test that action handler reuses loaded action."""
        config = ActionHookConfig(action_type="test_action", action_config={})
        handler = ActionHookHandler(config)

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(return_value=None)
        mock_action = Mock()
        mock_action.connector = mock_connector
        handler.action = mock_action  # Pre-load the action

        with patch("actions.load_action") as mock_load_action:
            result = await handler.execute(sample_context)
            assert result is True
            mock_load_action.assert_not_called()
            mock_connector.connect.assert_called_once()


class TestCreateHookHandler:
    """Test cases for create_hook_handler function."""

    def test_create_message_handler(self, sample_message_hook):
        """Test creating message hook handler."""
        handler = create_hook_handler(sample_message_hook)
        assert isinstance(handler, MessageHookHandler)
        assert isinstance(handler.config, MessageHookConfig)
        assert handler.config.message == sample_message_hook.handler_config["message"]

    def test_create_command_handler(self, sample_command_hook):
        """Test creating command hook handler."""
        handler = create_hook_handler(sample_command_hook)
        assert isinstance(handler, CommandHookHandler)
        assert isinstance(handler.config, CommandHookConfig)
        assert handler.config.command == sample_command_hook.handler_config["command"]

    def test_create_function_handler(self, sample_function_hook):
        """Test creating function hook handler."""
        handler = create_hook_handler(sample_function_hook)
        assert isinstance(handler, FunctionHookHandler)
        assert isinstance(handler.config, FunctionHookConfig)
        assert (
            handler.config.function == sample_function_hook.handler_config["function"]
        )
        assert (
            handler.config.module_name
            == sample_function_hook.handler_config["module_name"]
        )

    def test_create_action_handler(self, sample_action_hook):
        """Test creating action hook handler."""
        handler = create_hook_handler(sample_action_hook)
        assert isinstance(handler, ActionHookHandler)
        assert isinstance(handler.config, ActionHookConfig)
        assert (
            handler.config.action_type
            == sample_action_hook.handler_config["action_type"]
        )
        assert (
            handler.config.action_config
            == sample_action_hook.handler_config["action_config"]
        )

    def test_create_handler_unknown_type(self):
        """Test creating handler with unknown type."""
        hook = LifecycleHook(
            hook_type=LifecycleHookType.ON_ENTRY,
            handler_type="unknown_type",
            handler_config={},
        )

        with patch("runtime.hook.logging") as mock_logging:
            handler = create_hook_handler(hook)
            assert handler is None
            mock_logging.error.assert_called_once_with(
                "Unknown hook handler type: unknown_type"
            )

    def test_create_handler_case_insensitive(self):
        """Test creating handler with case-insensitive type."""
        hook = LifecycleHook(
            hook_type=LifecycleHookType.ON_ENTRY,
            handler_type="MESSAGE",  # Uppercase
            handler_config={"message": "test"},
        )

        handler = create_hook_handler(hook)
        assert isinstance(handler, MessageHookHandler)


class TestParseLifecycleHooks:
    """Test cases for parse_lifecycle_hooks function."""

    def test_parse_empty_hooks(self):
        """Test parsing empty hooks list."""
        result = parse_lifecycle_hooks([])
        assert result == []

    def test_parse_valid_hooks(self):
        """Test parsing valid hooks configuration."""
        raw_hooks = [
            {
                "hook_type": "on_entry",
                "handler_type": "message",
                "handler_config": {"message": "test"},
                "priority": 1,
            },
            {
                "hook_type": "on_exit",
                "handler_type": "command",
                "handler_config": {"command": "echo test"},
                "async_execution": False,
                "timeout_seconds": 10.0,
            },
        ]

        hooks = parse_lifecycle_hooks(raw_hooks)
        assert len(hooks) == 2

        assert hooks[0].hook_type == LifecycleHookType.ON_ENTRY
        assert hooks[0].handler_type == "message"
        assert hooks[0].priority == 1

        assert hooks[1].hook_type == LifecycleHookType.ON_EXIT
        assert hooks[1].handler_type == "command"
        assert hooks[1].async_execution is False
        assert hooks[1].timeout_seconds == 10.0

    def test_parse_hooks_with_defaults(self):
        """Test parsing hooks with default values."""
        raw_hooks = [
            {
                "hook_type": "on_startup",
                "handler_type": "function",
                "handler_config": {"function": "test"},
            }
        ]

        hooks = parse_lifecycle_hooks(raw_hooks)
        assert len(hooks) == 1

        hook = hooks[0]
        assert hook.async_execution is True  # Default
        assert hook.timeout_seconds == 5.0  # Default
        assert hook.on_failure == "ignore"  # Default
        assert hook.priority == 0  # Default

    def test_parse_hooks_invalid_hook_type(self):
        """Test parsing hooks with invalid hook type."""
        raw_hooks = [
            {
                "hook_type": "invalid_type",
                "handler_type": "message",
                "handler_config": {"message": "test"},
            }
        ]

        with patch("runtime.hook.logging") as mock_logging:
            hooks = parse_lifecycle_hooks(raw_hooks)
            assert len(hooks) == 0
            mock_logging.error.assert_called_once()

    def test_parse_hooks_missing_required_fields(self):
        """Test parsing hooks with missing required fields."""
        raw_hooks = [
            {
                "handler_type": "message",  # Missing hook_type
                "handler_config": {"message": "test"},
            },
            {
                "hook_type": "on_entry",  # Missing handler_type
                "handler_config": {"message": "test"},
            },
        ]

        with patch("runtime.hook.logging") as mock_logging:
            hooks = parse_lifecycle_hooks(raw_hooks)
            assert len(hooks) == 0
            assert mock_logging.error.call_count == 2


class TestExecuteLifecycleHooks:
    """Test cases for execute_lifecycle_hooks function."""

    @pytest.mark.asyncio
    async def test_execute_hooks_empty_list(self):
        """Test executing empty hooks list."""
        result = await execute_lifecycle_hooks([], LifecycleHookType.ON_ENTRY)
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_hooks_no_matching_type(self, sample_message_hook):
        """Test executing hooks with no matching type."""
        hooks = [sample_message_hook]  # ON_ENTRY hook
        result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_EXIT)
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_hooks_successful(self, sample_context):
        """Test successful execution of matching hooks."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "Hook 1"},
                priority=1,
            ),
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "Hook 2"},
                priority=2,
            ),
        ]

        with patch("runtime.hook.create_hook_handler") as mock_create:
            mock_handler1 = AsyncMock()
            mock_handler1.execute.return_value = True
            mock_handler2 = AsyncMock()
            mock_handler2.execute.return_value = True
            mock_create.side_effect = [mock_handler1, mock_handler2]

            with patch("runtime.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY, sample_context
                )
                assert result is True
                mock_logging.info.assert_called_once_with("Executing 2 on_entry hooks")

                assert mock_create.call_count == 2
                mock_handler2.execute.assert_called_once()  # Priority 2 first
                mock_handler1.execute.assert_called_once()  # Priority 1 second

    @pytest.mark.asyncio
    async def test_execute_hooks_priority_sorting(self):
        """Test that hooks are executed in priority order."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "Low priority"},
                priority=1,
            ),
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "High priority"},
                priority=5,
            ),
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "Medium priority"},
                priority=3,
            ),
        ]

        execution_order = []

        def track_execution(hook):
            handler = AsyncMock()

            async def execute_with_tracking(context):
                execution_order.append(hook.handler_config["message"])
                return True

            handler.execute = execute_with_tracking
            return handler

        with patch("runtime.hook.create_hook_handler", side_effect=track_execution):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is True
            assert execution_order == [
                "High priority",
                "Medium priority",
                "Low priority",
            ]

    @pytest.mark.asyncio
    async def test_execute_hooks_handler_creation_failure(self):
        """Test execution when handler creation fails."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
            )
        ]

        with patch("runtime.hook.create_hook_handler", return_value=None):
            with patch("runtime.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                mock_logging.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_hooks_failure_ignore_policy(self):
        """Test execution with failure ignore policy."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                on_failure="ignore",
            )
        ]

        mock_handler = AsyncMock()
        mock_handler.execute.return_value = False

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is False  # Overall result is False, but execution continues

    @pytest.mark.asyncio
    async def test_execute_hooks_failure_abort_policy(self):
        """Test execution with failure abort policy."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test1"},
                on_failure="abort",
            ),
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test2"},
            ),
        ]

        mock_handler1 = AsyncMock()
        mock_handler1.execute.return_value = False
        mock_handler2 = AsyncMock()
        mock_handler2.execute.return_value = True

        with patch(
            "runtime.hook.create_hook_handler",
            side_effect=[mock_handler1, mock_handler2],
        ):
            with patch("runtime.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                mock_logging.error.assert_called_once()
                mock_handler1.execute.assert_called_once()
                mock_handler2.execute.assert_not_called()  # Should not execute due to abort

    @pytest.mark.asyncio
    async def test_execute_hooks_timeout(self):
        """Test execution with timeout."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=0.1,
            )
        ]

        async def slow_execution(context):
            await asyncio.sleep(1)  # Takes longer than timeout
            return True

        mock_handler = AsyncMock()
        mock_handler.execute.side_effect = slow_execution

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            with patch("runtime.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                mock_logging.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_hooks_timeout_abort_policy(self):
        """Test execution timeout with abort policy."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=0.1,
                on_failure="abort",
            )
        ]

        async def slow_execution(context):
            await asyncio.sleep(1)
            return True

        mock_handler = AsyncMock()
        mock_handler.execute.side_effect = slow_execution

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is False

    @pytest.mark.asyncio
    async def test_execute_hooks_no_timeout(self):
        """Test execution with no timeout specified."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                timeout_seconds=None,
            )
        ]

        mock_handler = AsyncMock()
        mock_handler.execute.return_value = True

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is True

    @pytest.mark.asyncio
    async def test_execute_hooks_context_update(self, sample_context):
        """Test that context is properly updated with hook_type."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
            )
        ]

        received_context = None

        mock_handler = AsyncMock()

        async def capture_context(context):
            nonlocal received_context
            received_context = context.copy()
            return True

        mock_handler.execute = capture_context

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            await execute_lifecycle_hooks(
                hooks, LifecycleHookType.ON_ENTRY, sample_context
            )

            assert received_context is not None
            assert received_context["hook_type"] == "on_entry"
            assert received_context["mode_name"] == "test_mode"

    @pytest.mark.asyncio
    async def test_execute_hooks_general_exception(self):
        """Test execution with general exception."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                on_failure="ignore",
            )
        ]

        mock_handler = AsyncMock()
        mock_handler.execute.side_effect = Exception("General error")

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            with patch("runtime.hook.logging") as mock_logging:
                result = await execute_lifecycle_hooks(
                    hooks, LifecycleHookType.ON_ENTRY
                )
                assert result is False
                mock_logging.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_hooks_general_exception_abort(self):
        """Test execution with general exception and abort policy."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "test"},
                on_failure="abort",
            )
        ]

        mock_handler = AsyncMock()
        mock_handler.execute.side_effect = Exception("General error")

        with patch("runtime.hook.create_hook_handler", return_value=mock_handler):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is False

    @pytest.mark.asyncio
    async def test_execute_hooks_mixed_success_failure(self):
        """Test execution with mixed success and failure."""
        hooks = [
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "success"},
                priority=2,
            ),
            LifecycleHook(
                hook_type=LifecycleHookType.ON_ENTRY,
                handler_type="message",
                handler_config={"message": "failure"},
                priority=1,
                on_failure="ignore",
            ),
        ]

        mock_handler_success = AsyncMock()
        mock_handler_success.execute.return_value = True
        mock_handler_failure = AsyncMock()
        mock_handler_failure.execute.return_value = False

        with patch(
            "runtime.hook.create_hook_handler",
            side_effect=[mock_handler_success, mock_handler_failure],
        ):
            result = await execute_lifecycle_hooks(hooks, LifecycleHookType.ON_ENTRY)
            assert result is False
