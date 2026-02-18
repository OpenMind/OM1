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
def mock_providers():
    """Mock all external providers used by the ElevenLabs greeting connector."""
    with (
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ElevenLabsTTSProvider"
        ) as mock_tts_cls,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.GreetingConversationStateMachineProvider"
        ) as mock_state_cls,
        patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.ContextProvider"
        ) as mock_ctx_cls,
    ):
        mock_tts = Mock()
        mock_state = Mock()
        mock_ctx = Mock()
        mock_tts_cls.return_value = mock_tts
        mock_state_cls.return_value = mock_state
        mock_ctx_cls.return_value = mock_ctx
        yield {
            "tts_cls": mock_tts_cls,
            "tts": mock_tts,
            "state_cls": mock_state_cls,
            "state": mock_state,
            "ctx_cls": mock_ctx_cls,
            "ctx": mock_ctx,
        }


@pytest.fixture
def make_connector(mock_providers):
    """Factory to create a GreetingConversationConnector with mocked providers."""

    def _make(**config_kwargs):
        config = SpeakElevenLabsTTSConfig(**config_kwargs)
        return GreetingConversationConnector(config)

    return _make


@pytest.fixture
def connector(make_connector):
    """Create a connector with default config."""
    return make_connector()


@pytest.fixture
def greeting_input():
    """Create a standard greeting conversation input."""
    return GreetingConversationInput(
        response="Hello! Nice to meet you.",
        conversation_state=InterfaceConversationState.CONVERSING,
        confidence=0.9,
        speech_clarity=0.85,
    )


class TestGreetingConversationElevenLabsConnector:
    """Test the ElevenLabs greeting conversation connector."""

    def test_init_default_config(self, connector, mock_providers):
        """Test initialization with default config creates providers and starts TTS."""
        mock_providers["tts_cls"].assert_called_once()
        mock_providers["tts"].start.assert_called_once()
        mock_providers["state_cls"].assert_called_once()
        mock_providers["ctx_cls"].assert_called_once()
        assert connector.tts_duration == 0.0

    def test_init_custom_config(self, make_connector, mock_providers):
        """Test initialization with custom config passes values to TTS provider."""
        connector = make_connector(
            elevenlabs_api_key="custom_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="pcm_16000",
        )
        call_kwargs = mock_providers["tts_cls"].call_args[1]
        assert call_kwargs["elevenlabs_api_key"] == "custom_key"
        assert call_kwargs["voice_id"] == "custom_voice"
        assert call_kwargs["model_id"] == "custom_model"
        assert call_kwargs["output_format"] == "pcm_16000"
        assert connector is not None

    def test_init_sets_conversing_state(self, mock_providers, make_connector):
        """Test initialization sets state machine to CONVERSING."""
        make_connector()
        mock_providers["state"].start_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_adds_pending_message(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect adds the response text as a pending TTS message."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING
        }
        await connector.connect(greeting_input)
        mock_providers["tts"].add_pending_message.assert_called_once_with(
            "Hello! Nice to meet you."
        )

    @pytest.mark.asyncio
    async def test_connect_estimates_tts_duration(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect estimates TTS duration based on word count."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING
        }
        await connector.connect(greeting_input)
        # "Hello! Nice to meet you." = 5 words -> (5/100) * 60 = 3.0 seconds + 5 seconds buffer = 8.0 seconds total
        assert connector.tts_duration == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_connect_processes_conversation(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect calls state machine process_conversation with llm_output."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING
        }
        await connector.connect(greeting_input)
        mock_providers["state"].process_conversation.assert_called_once_with(
            {
                "conversation_state": InterfaceConversationState.CONVERSING,
                "response": "Hello! Nice to meet you.",
                "confidence": 0.9,
                "speech_clarity": 0.85,
            }
        )

    @pytest.mark.asyncio
    async def test_connect_finished_updates_context(self, connector, mock_providers):
        """Test connect updates context when conversation finishes."""
        finished_input = GreetingConversationInput(
            response="Goodbye!",
            conversation_state=InterfaceConversationState.FINISHED,
            confidence=0.95,
            speech_clarity=0.9,
        )
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.FINISHED
        }
        await connector.connect(finished_input)
        mock_providers["ctx"].update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )

    @pytest.mark.asyncio
    async def test_connect_not_finished_no_context_update(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect does not update context when conversation is not finished."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING
        }
        await connector.connect(greeting_input)
        mock_providers["ctx"].update_context.assert_not_called()

    def test_tick_skips_during_tts_activity(self, connector, mock_providers):
        """Test tick skips state update when TTS is still active."""
        connector.tts_triggered_time = float("inf")
        connector.tts_duration = 10.0
        with patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.logging"
        ):
            connector.tick()
        mock_providers["state"].update_state_without_llm.assert_not_called()

    def test_tick_updates_state_when_tts_idle(self, connector, mock_providers):
        """Test tick updates state when TTS is no longer active."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": "conversing",
            "confidence": {"overall": 0.8},
            "silence_duration": 2.0,
        }
        with patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.logging"
        ):
            connector.tick()
        mock_providers["state"].update_state_without_llm.assert_called_once()

    def test_tick_finished_updates_context(self, connector, mock_providers):
        """Test tick updates context when state machine detects conversation finished."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.FINISHED.value,
            "confidence": {"overall": 0.9},
            "silence_duration": 5.0,
        }
        with patch(
            "actions.greeting_conversation.connector.greeting_conversation_elevenlabs.logging"
        ):
            connector.tick()
        mock_providers["ctx"].update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )
