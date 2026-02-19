from unittest.mock import Mock, patch

import pytest

from hooks.greeting_hook import GeetingEndHookContext, geeting_end_hook


class TestGeetingEndHookContext:
    """Tests for GeetingEndHookContext model."""

    def test_context_default_values(self):
        """Test context model with default values."""
        context = GeetingEndHookContext()
        assert context.message == ""
        assert context.tts_provider == "elevenlabs"
        assert context.base_url is None
        assert context.api_key is None
        assert context.elevenlabs_api_key is None
        assert context.voice_id is None
        assert context.model_id is None
        assert context.output_format is None
        assert context.rate is None
        assert context.enable_tts_interrupt is False

    def test_context_custom_values(self):
        """Test context model with custom values."""
        context = GeetingEndHookContext(
            message="Hello, world!",
            tts_provider="kokoro",
            base_url="http://localhost:8880",
            api_key="test-api-key",
            elevenlabs_api_key="test-elevenlabs-key",
            voice_id="test-voice",
            model_id="test-model",
            output_format="mp3",
            rate=24000,
            enable_tts_interrupt=True,
        )
        assert context.message == "Hello, world!"
        assert context.tts_provider == "kokoro"
        assert context.base_url == "http://localhost:8880"
        assert context.api_key == "test-api-key"
        assert context.elevenlabs_api_key == "test-elevenlabs-key"
        assert context.voice_id == "test-voice"
        assert context.model_id == "test-model"
        assert context.output_format == "mp3"
        assert context.rate == 24000
        assert context.enable_tts_interrupt is True

    def test_context_extra_fields_allowed(self):
        """Test that extra fields are allowed in the context."""
        context = GeetingEndHookContext(
            message="Test", extra_field="extra_value"  # type: ignore
        )
        # Should not raise an error due to Config.extra = "allow"
        assert context.message == "Test"


@pytest.fixture
def mock_elevenlabs_provider():
    """Mock ElevenLabsTTSProvider."""
    with patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.start = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_kokoro_provider():
    """Mock KokoroTTSProvider."""
    with patch("hooks.greeting_hook.KokoroTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.start = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_riva_provider():
    """Mock RivaTTSProvider."""
    with patch("hooks.greeting_hook.RivaTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.start = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_greeting_state_provider():
    """Mock GreetingConversationStateMachineProvider."""
    with patch("hooks.greeting_hook.GreetingConversationStateMachineProvider") as mock:
        provider_instance = Mock()
        provider_instance.turn_count = 0
        provider_instance.max_turn_count = 5
        mock.return_value = provider_instance
        yield provider_instance


class TestGeetingEndHook:
    """Tests for geeting_end_hook function."""

    @pytest.mark.asyncio
    async def test_hook_with_elevenlabs_default_params(
        self, mock_elevenlabs_provider, mock_greeting_state_provider
    ):
        """Test hook with ElevenLabs provider and default parameters."""
        context = {"tts_provider": "elevenlabs"}
        mock_greeting_state_provider.turn_count = 3
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_elevenlabs_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            mock_provider_class.assert_called_once()
            call_kwargs = mock_provider_class.call_args[1]
            assert (
                call_kwargs["url"] == "https://api.openmind.org/api/core/elevenlabs/tts"
            )
            assert call_kwargs["voice_id"] == "JBFqnCBsd6RMkjVDRZzb"
            assert call_kwargs["model_id"] == "eleven_flash_v2_5"
            assert call_kwargs["output_format"] == "pcm_16000"
            assert call_kwargs["rate"] == 16000
            assert call_kwargs["enable_tts_interrupt"] is False

            mock_elevenlabs_provider.start.assert_called_once()
            mock_elevenlabs_provider.add_pending_message.assert_called_once()
            message = mock_elevenlabs_provider.add_pending_message.call_args[0][0]
            assert "nice talking to you" in message.lower()

    @pytest.mark.asyncio
    async def test_hook_with_elevenlabs_custom_params(
        self, mock_elevenlabs_provider, mock_greeting_state_provider
    ):
        """Test hook with ElevenLabs provider and custom parameters."""
        context = {
            "tts_provider": "elevenlabs",
            "base_url": "https://custom.api.com",
            "api_key": "test-api-key",
            "elevenlabs_api_key": "test-elevenlabs-key",
            "voice_id": "custom-voice",
            "model_id": "custom-model",
            "output_format": "mp3",
            "rate": 22050,
            "enable_tts_interrupt": True,
        }
        mock_greeting_state_provider.turn_count = 1
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_elevenlabs_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["url"] == "https://custom.api.com"
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["elevenlabs_api_key"] == "test-elevenlabs-key"
            assert call_kwargs["voice_id"] == "custom-voice"
            assert call_kwargs["model_id"] == "custom-model"
            assert call_kwargs["output_format"] == "mp3"
            assert call_kwargs["rate"] == 22050
            assert call_kwargs["enable_tts_interrupt"] is True

    @pytest.mark.asyncio
    async def test_hook_with_kokoro_default_params(
        self, mock_kokoro_provider, mock_greeting_state_provider
    ):
        """Test hook with Kokoro provider and default parameters."""
        context = {"tts_provider": "kokoro"}
        mock_greeting_state_provider.turn_count = 2
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.KokoroTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_kokoro_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["url"] == "http://127.0.0.1:8880/v1"
            assert call_kwargs["voice_id"] == "af_bella"
            assert call_kwargs["model_id"] == "kokoro"
            assert call_kwargs["output_format"] == "pcm"
            assert call_kwargs["rate"] == 24000
            assert call_kwargs["enable_tts_interrupt"] is False

            mock_kokoro_provider.start.assert_called_once()
            mock_kokoro_provider.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_with_kokoro_custom_params(
        self, mock_kokoro_provider, mock_greeting_state_provider
    ):
        """Test hook with Kokoro provider and custom parameters."""
        context = {
            "tts_provider": "Kokoro",  # Test case-insensitive
            "base_url": "http://localhost:9000",
            "api_key": "kokoro-api-key",
            "voice_id": "custom-kokoro-voice",
            "model_id": "kokoro-v2",
            "output_format": "wav",
            "rate": 48000,
            "enable_tts_interrupt": True,
        }
        mock_greeting_state_provider.turn_count = 1

        with (
            patch("hooks.greeting_hook.KokoroTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_kokoro_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["url"] == "http://localhost:9000"
            assert call_kwargs["api_key"] == "kokoro-api-key"
            assert call_kwargs["voice_id"] == "custom-kokoro-voice"
            assert call_kwargs["model_id"] == "kokoro-v2"
            assert call_kwargs["output_format"] == "wav"
            assert call_kwargs["rate"] == 48000
            assert call_kwargs["enable_tts_interrupt"] is True

    @pytest.mark.asyncio
    async def test_hook_with_riva_default_params(
        self, mock_riva_provider, mock_greeting_state_provider
    ):
        """Test hook with Riva provider and default parameters."""
        context = {"tts_provider": "riva"}
        mock_greeting_state_provider.turn_count = 1
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.RivaTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_riva_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["url"] == "http://127.0.0.1:50051"

            mock_riva_provider.start.assert_called_once()
            mock_riva_provider.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_with_riva_custom_params(
        self, mock_riva_provider, mock_greeting_state_provider
    ):
        """Test hook with Riva provider and custom parameters."""
        context = {
            "tts_provider": "RIVA",  # Test case-insensitive
            "base_url": "http://riva-server:8000",
            "api_key": "riva-api-key",
        }
        mock_greeting_state_provider.turn_count = 2

        with (
            patch("hooks.greeting_hook.RivaTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_riva_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["url"] == "http://riva-server:8000"
            assert call_kwargs["api_key"] == "riva-api-key"

    @pytest.mark.asyncio
    async def test_hook_max_turn_count_reached(
        self, mock_elevenlabs_provider, mock_greeting_state_provider
    ):
        """Test hook when max turn count is reached."""
        context = {"tts_provider": "elevenlabs"}
        mock_greeting_state_provider.turn_count = 5
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_elevenlabs_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            mock_elevenlabs_provider.start.assert_called_once()
            mock_elevenlabs_provider.add_pending_message.assert_called_once()

            message = mock_elevenlabs_provider.add_pending_message.call_args[0][0]
            assert "enjoyed our conversation" in message.lower()
            assert "need to help others" in message.lower()

    @pytest.mark.asyncio
    async def test_hook_turn_count_exceeded(
        self, mock_kokoro_provider, mock_greeting_state_provider
    ):
        """Test hook when turn count exceeds max."""
        context = {"tts_provider": "kokoro"}
        mock_greeting_state_provider.turn_count = 10
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.KokoroTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_kokoro_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            message = mock_kokoro_provider.add_pending_message.call_args[0][0]
            assert "enjoyed our conversation" in message.lower()

    @pytest.mark.asyncio
    async def test_hook_zero_turn_count(
        self, mock_riva_provider, mock_greeting_state_provider
    ):
        """Test hook when turn count is zero."""
        context = {"tts_provider": "riva"}
        mock_greeting_state_provider.turn_count = 0
        mock_greeting_state_provider.max_turn_count = 5

        with (
            patch("hooks.greeting_hook.RivaTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_riva_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            mock_riva_provider.start.assert_called_once()
            message = mock_riva_provider.add_pending_message.call_args[0][0]
            assert "no worries" in message.lower()

    @pytest.mark.asyncio
    async def test_hook_unsupported_provider(self, mock_greeting_state_provider):
        """Test hook with unsupported TTS provider."""
        context = {"tts_provider": "unsupported_provider"}
        mock_greeting_state_provider.turn_count = 1

        with patch(
            "hooks.greeting_hook.GreetingConversationStateMachineProvider"
        ) as mock_state_class:
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

    @pytest.mark.asyncio
    async def test_hook_provider_initialization_error(
        self, mock_greeting_state_provider
    ):
        """Test hook when provider initialization fails."""
        context = {"tts_provider": "elevenlabs"}
        mock_greeting_state_provider.turn_count = 1

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.side_effect = Exception(
                "Provider initialization failed"
            )
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

    @pytest.mark.asyncio
    async def test_hook_provider_start_error(
        self, mock_elevenlabs_provider, mock_greeting_state_provider
    ):
        """Test hook when provider.start() fails."""
        context = {"tts_provider": "elevenlabs"}
        mock_greeting_state_provider.turn_count = 1
        mock_elevenlabs_provider.start.side_effect = Exception("Start failed")

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_elevenlabs_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

    @pytest.mark.asyncio
    async def test_hook_empty_context(
        self, mock_elevenlabs_provider, mock_greeting_state_provider
    ):
        """Test hook with empty context defaults to elevenlabs."""
        context = {}
        mock_greeting_state_provider.turn_count = 1

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.return_value = mock_elevenlabs_provider
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            mock_provider_class.assert_called_once()
            mock_elevenlabs_provider.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_error_logging(self, mock_greeting_state_provider, caplog):
        """Test that errors are logged properly."""
        context = {"tts_provider": "elevenlabs"}
        mock_greeting_state_provider.turn_count = 1

        with (
            patch("hooks.greeting_hook.ElevenLabsTTSProvider") as mock_provider_class,
            patch(
                "hooks.greeting_hook.GreetingConversationStateMachineProvider"
            ) as mock_state_class,
        ):
            mock_provider_class.side_effect = ValueError("Test error")
            mock_state_class.return_value = mock_greeting_state_provider

            await geeting_end_hook(context)

            assert "Error in geeting_end_hook" in caplog.text
            assert "Test error" in caplog.text
