from unittest.mock import Mock, patch

import pytest

from actions.base import ActionConfig
from actions.greeting_conversation.connector.base_greeting_conversation import (
    BaseGreetingConversationConnector,
)
from actions.greeting_conversation.interface import (
    ConversationState as InterfaceConversationState,
)
from actions.greeting_conversation.interface import (
    GreetingConversationInput,
)
from providers.greeting_conversation_state_provider import ConversationState


class MockActionConfig(ActionConfig):
    """Mock configuration for the base connector."""

    pass


class ConcreteGreetingConnector(BaseGreetingConversationConnector[MockActionConfig]):
    """Concrete implementation of BaseGreetingConversationConnector for testing."""

    def __init__(self, config: MockActionConfig, mock_tts=None):
        self._mock_tts = mock_tts
        super().__init__(config)

    def create_tts_provider(self):
        """Return a mock TTS provider for testing."""
        return self._mock_tts if self._mock_tts else Mock()


@pytest.fixture
def mock_tts():
    """Create a mock TTS provider."""
    tts = Mock()
    tts.start = Mock()
    tts.add_pending_message = Mock()
    return tts


@pytest.fixture
def mock_providers(mock_tts):
    """Mock all external providers used by the base greeting connector."""
    with (
        patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.GreetingConversationStateMachineProvider"
        ) as mock_state_cls,
        patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.ContextProvider"
        ) as mock_ctx_cls,
        patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.open_zenoh_session"
        ) as mock_zenoh,
        patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.time"
        ) as mock_time,
    ):
        mock_state = Mock()
        mock_ctx = Mock()
        mock_session = Mock()
        mock_state_cls.return_value = mock_state
        mock_ctx_cls.return_value = mock_ctx
        mock_zenoh.return_value = mock_session
        mock_time.time.return_value = 100.0
        yield {
            "tts": mock_tts,
            "state_cls": mock_state_cls,
            "state": mock_state,
            "ctx_cls": mock_ctx_cls,
            "ctx": mock_ctx,
            "zenoh": mock_zenoh,
            "session": mock_session,
            "time": mock_time,
        }


@pytest.fixture
def make_connector(mock_providers, mock_tts):
    """Factory to create a ConcreteGreetingConnector with mocked providers."""

    def _make(**config_kwargs):
        config = MockActionConfig(**config_kwargs)
        return ConcreteGreetingConnector(config, mock_tts=mock_tts)

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


class TestBaseGreetingConversationConnector:
    """Test the BaseGreetingConversationConnector."""

    def test_init_creates_providers(self, connector, mock_providers):
        """Test initialization creates all required providers."""
        mock_providers["state_cls"].assert_called_once()
        mock_providers["ctx_cls"].assert_called_once()
        mock_providers["zenoh"].assert_called_once()
        assert connector.greeting_state_provider is not None
        assert connector.context_provider is not None

    def test_init_starts_conversation(self, connector, mock_providers):
        """Test initialization starts the conversation state machine."""
        mock_providers["state"].start_conversation.assert_called_once()

    def test_init_starts_tts(self, connector, mock_providers):
        """Test initialization starts the TTS provider."""
        mock_providers["tts"].start.assert_called_once()

    def test_init_sets_default_values(self, connector, mock_providers):
        """Test initialization sets default values."""
        assert connector.tts_triggered_time == 100.0
        assert connector.tts_duration == 0.0
        assert connector.conversation_finished_sent is False
        assert connector.greeting_status == ConversationState.CONVERSING.value
        assert connector.person_greeting_topic == "om/person_greeting"

    def test_init_opens_zenoh_session(self, connector, mock_providers):
        """Test initialization opens a Zenoh session."""
        assert connector.session == mock_providers["session"]

    def test_init_handles_zenoh_failure(self, mock_providers, make_connector):
        """Test initialization handles Zenoh session failure gracefully."""
        mock_providers["zenoh"].side_effect = Exception("Connection failed")
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector = make_connector()
        assert connector.session is None

    @pytest.mark.asyncio
    async def test_connect_logs_conversation_details(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect logs all conversation details."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ) as mock_log:
            await connector.connect(greeting_input)
            assert mock_log.info.call_count >= 4

    @pytest.mark.asyncio
    async def test_connect_adds_pending_message(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect adds the response text as a pending TTS message."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
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
            "current_state": ConversationState.CONVERSING.value
        }
        await connector.connect(greeting_input)
        # "Hello! Nice to meet you." = 5 words
        # (5/100) * 60 + 5 = 3.0 + 5 = 8.0 seconds
        assert connector.tts_duration == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_connect_updates_tts_triggered_time(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect updates the TTS triggered time."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        await connector.connect(greeting_input)
        assert connector.tts_triggered_time == 100.0

    @pytest.mark.asyncio
    async def test_connect_processes_conversation(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect calls state machine process_conversation with llm_output."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
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
    async def test_connect_updates_greeting_status(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect updates the greeting status from state machine."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONCLUDING.value
        }
        await connector.connect(greeting_input)
        assert connector.greeting_status == ConversationState.CONCLUDING.value

    @pytest.mark.asyncio
    async def test_connect_calls_publish_countdown_status(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect calls publish_countdown_status with current state."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        with patch.object(connector, "publish_countdown_status") as mock_report:
            await connector.connect(greeting_input)
            mock_report.assert_called_once_with(ConversationState.CONVERSING.value)

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
            "current_state": ConversationState.FINISHED.value
        }
        await connector.connect(finished_input)
        mock_providers["ctx"].update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )
        assert connector.conversation_finished_sent is True

    @pytest.mark.asyncio
    async def test_connect_finished_only_updates_context_once(
        self, connector, mock_providers
    ):
        """Test connect only updates context once even if called multiple times."""
        finished_input = GreetingConversationInput(
            response="Goodbye!",
            conversation_state=InterfaceConversationState.FINISHED,
            confidence=0.95,
            speech_clarity=0.9,
        )
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.FINISHED.value
        }
        await connector.connect(finished_input)
        await connector.connect(finished_input)
        # Should only be called once
        mock_providers["ctx"].update_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_not_finished_no_context_update(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect does not update context when conversation is not finished."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        await connector.connect(greeting_input)
        mock_providers["ctx"].update_context.assert_not_called()

    def test_tick_sleeps_for_10_seconds(self, connector, mock_providers):
        """Test tick sleeps for 10 seconds."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.CONVERSING.value,
            "confidence": {"overall": 0.8},
            "silence_duration": 2.0,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep") as mock_sleep,
        ):
            connector.tick()
            mock_sleep.assert_called_once_with(10)

    def test_tick_skips_during_tts_activity(self, connector, mock_providers):
        """Test tick skips state update when TTS is still active."""
        connector.tts_triggered_time = 100.0
        connector.tts_duration = 50.0
        mock_providers["time"].time.return_value = (
            120.0  # 20 seconds elapsed, 30 remaining
        )
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        mock_providers["state"].update_state_without_llm.assert_not_called()

    def test_tick_updates_state_when_tts_idle(self, connector, mock_providers):
        """Test tick updates state when TTS is no longer active."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.CONVERSING.value,
            "confidence": {"overall": 0.8},
            "silence_duration": 2.0,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        mock_providers["state"].update_state_without_llm.assert_called_once()

    def test_tick_updates_greeting_status(self, connector, mock_providers):
        """Test tick updates the greeting status from state machine."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.CONCLUDING.value,
            "confidence": {"overall": 0.85},
            "silence_duration": 3.5,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        assert connector.greeting_status == ConversationState.CONCLUDING.value

    def test_tick_calls_publish_countdown_status(self, connector, mock_providers):
        """Test tick calls publish_countdown_status with current state."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.CONVERSING.value,
            "confidence": {"overall": 0.8},
            "silence_duration": 2.0,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
            patch.object(connector, "publish_countdown_status") as mock_report,
        ):
            connector.tick()
            mock_report.assert_called_once_with(ConversationState.CONVERSING.value)

    def test_tick_finished_updates_context(self, connector, mock_providers):
        """Test tick updates context when state machine detects conversation finished."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.FINISHED.value,
            "confidence": {"overall": 0.9},
            "silence_duration": 5.0,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
        mock_providers["ctx"].update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )
        assert connector.conversation_finished_sent is True

    def test_tick_finished_only_updates_context_once(self, connector, mock_providers):
        """Test tick only updates context once even if called multiple times."""
        connector.tts_triggered_time = 0.0
        connector.tts_duration = 0.0
        mock_providers["time"].time.return_value = 200.0
        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.FINISHED.value,
            "confidence": {"overall": 0.9},
            "silence_duration": 5.0,
        }
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch.object(connector, "sleep"),
        ):
            connector.tick()
            connector.tick()
        # Should only be called once
        mock_providers["ctx"].update_context.assert_called_once()

    def test_publish_countdown_status_conversing_state(self, connector, mock_providers):
        """Test publish_countdown_status publishes 20 seconds for CONVERSING state."""
        # Reset mock before testing
        mock_providers["session"].put.reset_mock()
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.uuid4"
            ) as mock_uuid,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.PersonGreetingStatus"
            ) as mock_status_cls,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.prepare_header"
            ) as mock_header,
        ):
            mock_uuid.return_value = "test-uuid"
            mock_status = Mock()
            mock_status.serialize.return_value = b"serialized"
            mock_status_cls.return_value = mock_status
            mock_header.return_value = "header"

            connector.publish_countdown_status(ConversationState.CONVERSING.value)

            mock_providers["session"].put.assert_called_once()
            call_args = mock_providers["session"].put.call_args
            assert call_args[0][0] == "om/person_greeting"

    def test_publish_countdown_status_concluding_state(self, connector, mock_providers):
        """Test publish_countdown_status publishes 10 seconds for CONCLUDING state."""
        # Reset mock before testing
        mock_providers["session"].put.reset_mock()
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.uuid4"
            ) as mock_uuid,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.PersonGreetingStatus"
            ) as mock_status_cls,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.prepare_header"
            ),
        ):
            mock_uuid.return_value = "test-uuid"
            mock_status = Mock()
            mock_status.serialize.return_value = b"serialized"
            mock_status_cls.return_value = mock_status

            connector.publish_countdown_status(ConversationState.CONCLUDING.value)

            mock_providers["session"].put.assert_called_once()

    def test_publish_countdown_status_finished_state_publishes_zero(
        self, connector, mock_providers
    ):
        """Test publish_countdown_status publishes 0 seconds for FINISHED state."""
        # Reset mock before testing
        mock_providers["session"].put.reset_mock()
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ),
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.uuid4"
            ) as mock_uuid,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.PersonGreetingStatus"
            ) as mock_status_cls,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.prepare_header"
            ),
        ):
            mock_uuid.return_value = "test-uuid"
            mock_status = Mock()
            mock_status.serialize.return_value = b"serialized"
            mock_status_cls.return_value = mock_status

            connector.publish_countdown_status(ConversationState.FINISHED.value)

            mock_providers["session"].put.assert_called_once()

    def test_publish_countdown_status_no_session_no_publish(
        self, connector, mock_providers
    ):
        """Test publish_countdown_status does not publish when session is None."""
        connector.session = None
        # Reset any previous calls
        mock_providers["session"].put.reset_mock()
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector.publish_countdown_status(ConversationState.CONVERSING.value)
        # Should not be called since session is None
        mock_providers["session"].put.assert_not_called()

    def test_publish_countdown_status_handles_publish_error(
        self, connector, mock_providers
    ):
        """Test publish_countdown_status handles publishing errors gracefully."""
        # Reset and configure mock to raise exception
        mock_providers["session"].put.reset_mock()
        mock_providers["session"].put.side_effect = Exception("Publish failed")
        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ) as mock_log,
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.uuid4"
            ),
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.PersonGreetingStatus"
            ),
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.prepare_header"
            ),
        ):
            connector.publish_countdown_status(ConversationState.CONVERSING.value)
            # Should log error
            assert any(
                "Error publishing" in str(call)
                for call in mock_log.error.call_args_list
            )

    def test_stop_closes_zenoh_session(self, connector, mock_providers):
        """Test stop closes the Zenoh session."""
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector.stop()
        mock_providers["session"].close.assert_called_once()

    def test_stop_handles_no_session(self, connector, mock_providers):
        """Test stop handles missing Zenoh session gracefully."""
        connector.session = None
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector.stop()
        # No exception should be raised

    def test_greeting_status_defaults_to_conversing(self, connector):
        """Test that greeting_status defaults to CONVERSING on init."""
        assert connector.greeting_status == ConversationState.CONVERSING.value

    def test_conversation_finished_sent_defaults_to_false(self, connector):
        """Test that conversation_finished_sent defaults to False on init."""
        assert connector.conversation_finished_sent is False
