import time
from unittest.mock import Mock, patch

import pytest

from actions.greeting_conversation.connector.greeting_conversation_kokoro import (
    GreetingConversationConnector,
    SpeakKokoroTTSConfig,
)
from actions.greeting_conversation.interface import (
    ConversationState as InterfaceConversationState,
)
from actions.greeting_conversation.interface import (
    GreetingConversationInput,
)
from providers.greeting_conversation_state_provider import ConversationState


@pytest.fixture
def mock_config():
    """Create a mock SpeakKokoroTTSConfig with default values."""
    config = SpeakKokoroTTSConfig(
        api_key="test_api_key",  # type: ignore
        voice_id="af_bella",
        model_id="kokoro",
        output_format="pcm",
        rate=24000,
        enable_tts_interrupt=False,
        silence_rate=0,
    )
    return config


@pytest.fixture
def mock_minimal_config():
    """Create a minimal config using defaults."""
    config = SpeakKokoroTTSConfig()
    return config


@pytest.fixture
def mock_config_with_tts_interrupt():
    """Create a config with TTS interrupt enabled."""
    config = SpeakKokoroTTSConfig(
        enable_tts_interrupt=True,
    )
    return config


@pytest.fixture
def greeting_input():
    """Create a GreetingConversationInput instance for testing."""
    return GreetingConversationInput(
        response="Hello! How can I help you today?",
        conversation_state=InterfaceConversationState.CONVERSING,
        confidence=0.85,
        speech_clarity=0.9,
    )


@pytest.fixture
def greeting_input_finished():
    """Create a GreetingConversationInput indicating finished conversation."""
    return GreetingConversationInput(
        response="Goodbye! Have a great day!",
        conversation_state=InterfaceConversationState.FINISHED,
        confidence=0.95,
        speech_clarity=0.88,
    )


@pytest.fixture
def greeting_input_long_response():
    """Create a GreetingConversationInput with a long response."""
    long_text = " ".join(["word"] * 200)  # 200 words
    return GreetingConversationInput(
        response=long_text,
        conversation_state=InterfaceConversationState.CONVERSING,
        confidence=0.9,
        speech_clarity=0.85,
    )


def test_init_with_full_config(mock_config):
    """Test initialization with full configuration."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ) as mock_context_provider_class,
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance
        mock_state_instance = Mock()
        mock_state_provider_class.return_value = mock_state_instance
        mock_context_instance = Mock()
        mock_context_provider_class.return_value = mock_context_instance

        connector = GreetingConversationConnector(mock_config)

        # Verify TTS provider was initialized correctly
        mock_tts_provider_class.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key="test_api_key",
            voice_id="af_bella",
            model_id="kokoro",
            output_format="pcm",
            rate=24000,
            enable_tts_interrupt=False,
        )

        mock_tts_instance.start.assert_called_once()
        mock_state_provider_class.assert_called_once()
        mock_context_provider_class.assert_called_once()

        assert connector.tts_duration == 0.0
        assert isinstance(connector.tts_triggered_time, float)
        assert (
            connector.greeting_state_provider.current_state
            == ConversationState.CONVERSING
        )


def test_init_with_minimal_config(mock_minimal_config):
    """Test initialization with minimal configuration using defaults."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        GreetingConversationConnector(mock_minimal_config)

        mock_tts_provider_class.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key=None,
            voice_id="af_bella",
            model_id="kokoro",
            output_format="pcm",
            rate=24000,
            enable_tts_interrupt=False,
        )

        mock_tts_instance.start.assert_called_once()


def test_init_with_tts_interrupt(mock_config_with_tts_interrupt):
    """Test initialization with TTS interrupt enabled."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        GreetingConversationConnector(mock_config_with_tts_interrupt)

        # Verify TTS interrupt was passed correctly
        call_args = mock_tts_provider_class.call_args
        assert call_args.kwargs["enable_tts_interrupt"] is True


@pytest.mark.asyncio
async def test_connect_basic(mock_config, greeting_input):
    """Test basic connect functionality."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.CONVERSING}
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        await connector.connect(greeting_input)

        mock_tts_instance.add_pending_message.assert_called_once_with(
            "Hello! How can I help you today?"
        )

        expected_llm_output = {
            "conversation_state": InterfaceConversationState.CONVERSING,
            "response": "Hello! How can I help you today?",
            "confidence": 0.85,
            "speech_clarity": 0.9,
        }
        mock_state_instance.process_conversation.assert_called_once_with(
            expected_llm_output
        )


@pytest.mark.asyncio
async def test_connect_tts_duration_calculation(mock_config, greeting_input):
    """Test that TTS duration is calculated based on text length."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.CONVERSING}
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        # Response has 7 words: "Hello! How can I help you today?"
        # Expected duration: (7 / 100.0) * 60.0 = 4.2 seconds
        start_time = time.time()
        await connector.connect(greeting_input)

        expected_duration = (7 / 100.0) * 60.0
        assert connector.tts_duration == expected_duration
        assert connector.tts_triggered_time >= start_time


@pytest.mark.asyncio
async def test_connect_finished_state(mock_config, greeting_input_finished):
    """Test connect when conversation reaches FINISHED state."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ) as mock_context_provider_class,
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.FINISHED}
        )
        mock_state_provider_class.return_value = mock_state_instance

        mock_context_instance = Mock()
        mock_context_instance.update_context = Mock()
        mock_context_provider_class.return_value = mock_context_instance

        connector = GreetingConversationConnector(mock_config)

        await connector.connect(greeting_input_finished)

        # Verify context was updated when conversation finished
        mock_context_instance.update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )


@pytest.mark.asyncio
async def test_connect_with_empty_response(mock_config):
    """Test connect with empty response string."""
    empty_input = GreetingConversationInput(
        response="",
        conversation_state=InterfaceConversationState.CONVERSING,
        confidence=0.5,
        speech_clarity=0.5,
    )

    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.CONVERSING}
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        await connector.connect(empty_input)

        # TTS should still be called even with empty string
        mock_tts_instance.add_pending_message.assert_called_once_with("")

        # Duration should be 0 for empty response
        assert connector.tts_duration == 0.0


@pytest.mark.asyncio
async def test_connect_with_long_response(mock_config, greeting_input_long_response):
    """Test connect with a long response to verify TTS duration calculation."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.CONVERSING}
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        await connector.connect(greeting_input_long_response)

        # 200 words: (200 / 100.0) * 60.0 = 120 seconds
        expected_duration = (200 / 100.0) * 60.0
        assert connector.tts_duration == expected_duration


@pytest.mark.asyncio
async def test_connect_updates_tts_triggered_time(mock_config, greeting_input):
    """Test that connect updates tts_triggered_time."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_instance.add_pending_message = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.process_conversation = Mock(
            return_value={"current_state": ConversationState.CONVERSING}
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        initial_time = connector.tts_triggered_time

        # Wait a bit to ensure time difference
        time.sleep(0.01)

        await connector.connect(greeting_input)

        # tts_triggered_time should be updated
        assert connector.tts_triggered_time > initial_time


def test_tick_skips_during_tts_playback(mock_config):
    """Test that tick skips state update when TTS is playing."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock()
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        # Simulate recent TTS activity
        connector.tts_triggered_time = time.time()
        connector.tts_duration = 10.0  # 10 seconds

        # Mock sleep to avoid actual delay
        with patch.object(connector, "sleep"):
            connector.tick()

        # Should not call update_state_without_llm during TTS playback
        mock_state_instance.update_state_without_llm.assert_not_called()


def test_tick_updates_state_after_tts_playback(mock_config):
    """Test that tick updates state after TTS playback finishes."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.CONVERSING.value,
                "confidence": {"overall": 0.75},
                "silence_duration": 5.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        # Simulate TTS activity that has finished
        connector.tts_triggered_time = time.time() - 20.0  # 20 seconds ago
        connector.tts_duration = 10.0  # 10 second duration

        # Mock sleep to avoid actual delay
        with patch.object(connector, "sleep"):
            connector.tick()

        # Should call update_state_without_llm after TTS playback
        mock_state_instance.update_state_without_llm.assert_called_once()


def test_tick_detects_finished_state(mock_config):
    """Test that tick detects finished conversation state."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ) as mock_context_provider_class,
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.FINISHED.value,
                "confidence": {"overall": 0.9},
                "silence_duration": 30.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        mock_context_instance = Mock()
        mock_context_instance.update_context = Mock()
        mock_context_provider_class.return_value = mock_context_instance

        connector = GreetingConversationConnector(mock_config)

        # Simulate TTS activity that has finished
        connector.tts_triggered_time = time.time() - 20.0
        connector.tts_duration = 10.0

        # Mock sleep to avoid actual delay
        with patch.object(connector, "sleep"):
            connector.tick()

        # Should update context when finished
        mock_context_instance.update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )


def test_tick_sleeps_for_10_seconds(mock_config):
    """Test that tick sleeps for 10 seconds."""
    with (
        patch("providers.kokoro_tts_provider.AudioOutputLiveStream"),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.KokoroTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_kokoro.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.CONVERSING.value,
                "confidence": {"overall": 0.75},
                "silence_duration": 5.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)

        # Simulate TTS activity that has finished
        connector.tts_triggered_time = time.time() - 20.0
        connector.tts_duration = 10.0

        # Mock sleep to verify it's called with correct duration
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(10)


def test_config_defaults():
    """Test that SpeakKokoroTTSConfig has correct default values."""
    config = SpeakKokoroTTSConfig()

    assert config.voice_id == "af_bella"
    assert config.model_id == "kokoro"
    assert config.output_format == "pcm"
    assert config.rate == 24000
    assert config.enable_tts_interrupt is False
    assert config.silence_rate == 0


def test_config_custom_values():
    """Test that SpeakKokoroTTSConfig accepts custom values."""
    config = SpeakKokoroTTSConfig(
        voice_id="custom_voice",
        model_id="custom_model",
        output_format="wav",
        rate=48000,
        enable_tts_interrupt=True,
        silence_rate=2,
    )

    assert config.voice_id == "custom_voice"
    assert config.model_id == "custom_model"
    assert config.output_format == "wav"
    assert config.rate == 48000
    assert config.enable_tts_interrupt is True
    assert config.silence_rate == 2
