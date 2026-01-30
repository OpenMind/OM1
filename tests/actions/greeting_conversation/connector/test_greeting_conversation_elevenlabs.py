import time
from unittest.mock import Mock, patch

import pytest

from actions.greeting_conversation.connector.greeting_conversation_elevenlabs import (
    GreetingConversationConnector,
    SpeakElevenLabsTTSConfig,
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
    """Create a mock SpeakElevenLabsTTSConfig with default values."""
    config = SpeakElevenLabsTTSConfig(
        api_key="test_api_key",  # type: ignore
        elevenlabs_api_key="test_elevenlabs_key",
        voice_id="test_voice_id",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        silence_rate=0,
    )
    return config


@pytest.fixture
def mock_minimal_config():
    """Create a minimal config using defaults."""
    config = SpeakElevenLabsTTSConfig()
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


def test_init_with_full_config(mock_config):
    """Test initialization with full configuration."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key="test_api_key",
            elevenlabs_api_key="test_elevenlabs_key",
            voice_id="test_voice_id",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        mock_tts_instance.start.assert_called_once()
        mock_state_provider_class.assert_called_once()
        mock_context_provider_class.assert_called_once()

        assert connector.tts_duration == 0.0
        assert isinstance(connector.tts_triggered_time, float)


def test_init_with_minimal_config(mock_minimal_config):
    """Test initialization with minimal configuration using defaults."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        GreetingConversationConnector(mock_minimal_config)

        mock_tts_provider_class.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=None,
            elevenlabs_api_key=None,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        mock_tts_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_connect_basic(mock_config, greeting_input):
    """Test basic connect functionality."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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
    """Test connect with empty response."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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

        empty_input = GreetingConversationInput(
            response="",
            conversation_state=InterfaceConversationState.CONVERSING,
            confidence=0.5,
            speech_clarity=0.6,
        )

        await connector.connect(empty_input)

        # Verify empty message was still added to TTS
        mock_tts_instance.add_pending_message.assert_called_once_with("")

        # Duration should be 0 for empty response
        assert connector.tts_duration == 0.0


def test_tick_basic(mock_config):
    """Test basic tick functionality."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.CONVERSING.value,
                "confidence": {"overall": 0.5},
                "silence_duration": 3.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)
        connector.sleep = Mock()  # Mock sleep to avoid actual delay

        connector.tts_duration = 0.0

        connector.tick()

        connector.sleep.assert_called_once_with(10)

        mock_state_instance.update_state_without_llm.assert_called_once()


def test_tick_skips_during_tts(mock_config):
    """Test that tick skips state update when TTS is still active."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock()
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)
        connector.sleep = Mock()  # Mock sleep to avoid actual delay

        connector.tts_triggered_time = time.time()
        connector.tts_duration = 100.0  # Large duration to ensure it's still active

        connector.tick()

        connector.sleep.assert_called_once_with(10)

        mock_state_instance.update_state_without_llm.assert_not_called()


def test_tick_finished_state(mock_config):
    """Test tick when conversation reaches FINISHED state."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ) as mock_context_provider_class,
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.FINISHED.value,
                "confidence": {"overall": 0.95},
                "silence_duration": 12.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        mock_context_instance = Mock()
        mock_context_instance.update_context = Mock()
        mock_context_provider_class.return_value = mock_context_instance

        connector = GreetingConversationConnector(mock_config)
        connector.sleep = Mock()  # Mock sleep to avoid actual delay

        connector.tts_duration = 0.0

        connector.tick()

        mock_context_instance.update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )


def test_config_values():
    """Test that config values are properly stored and used."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ),
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ),
    ):
        custom_config = SpeakElevenLabsTTSConfig(
            api_key="custom_api_key",  # type: ignore
            elevenlabs_api_key="custom_elevenlabs_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="custom_format",
            silence_rate=5,
        )

        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        GreetingConversationConnector(custom_config)

        mock_tts_provider_class.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key="custom_api_key",
            elevenlabs_api_key="custom_elevenlabs_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="custom_format",
        )


def test_config_default_values():
    """Test that config has correct default values."""
    config = SpeakElevenLabsTTSConfig()

    assert config.elevenlabs_api_key is None
    assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
    assert config.model_id == "eleven_flash_v2_5"
    assert config.output_format == "mp3_44100_128"
    assert config.silence_rate == 0


@pytest.mark.asyncio
async def test_connect_long_response(mock_config):
    """Test connect with a long response to verify TTS duration calculation."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
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

        long_response = " ".join(["word"] * 100)
        long_input = GreetingConversationInput(
            response=long_response,
            conversation_state=InterfaceConversationState.CONVERSING,
            confidence=0.85,
            speech_clarity=0.9,
        )

        await connector.connect(long_input)

        expected_duration = 60.0
        assert connector.tts_duration == expected_duration


def test_tick_after_tts_completion(mock_config):
    """Test tick behavior after TTS has completed."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ),
    ):
        mock_tts_instance = Mock()
        mock_tts_provider_class.return_value = mock_tts_instance

        mock_state_instance = Mock()
        mock_state_instance.update_state_without_llm = Mock(
            return_value={
                "current_state": ConversationState.CONVERSING.value,
                "confidence": {"overall": 0.6},
                "silence_duration": 7.0,
            }
        )
        mock_state_provider_class.return_value = mock_state_instance

        connector = GreetingConversationConnector(mock_config)
        connector.sleep = Mock()

        connector.tts_triggered_time = time.time() - 10.0
        connector.tts_duration = 5.0

        connector.tick()

        mock_state_instance.update_state_without_llm.assert_called_once()
