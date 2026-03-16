import time
from unittest.mock import Mock, patch

import pytest

from actions.base import ActionConfig
from actions.greeting_conversation.connector.base_greeting_conversation import (
    BaseGreetingConversationConnector,
    normalize_tts_text,
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
    tts.create_pending_message = Mock(return_value={"text": "mock text"})
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
    ):
        mock_state = Mock()
        mock_ctx = Mock()
        mock_session = Mock()
        mock_audio_pub = Mock()
        mock_state_cls.return_value = mock_state
        mock_ctx_cls.return_value = mock_ctx
        mock_zenoh.return_value = mock_session
        mock_session.declare_publisher.return_value = mock_audio_pub
        yield {
            "tts": mock_tts,
            "state_cls": mock_state_cls,
            "state": mock_state,
            "ctx_cls": mock_ctx_cls,
            "ctx": mock_ctx,
            "zenoh": mock_zenoh,
            "session": mock_session,
            "audio_pub": mock_audio_pub,
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
        assert connector.tts_request_id is None
        assert connector.tts_playing is False
        assert connector.conversation_finished_sent is False
        assert connector.greeting_status == ConversationState.CONVERSING.value
        assert connector.person_greeting_topic == "om/person_greeting"
        assert connector.audio_topic == "robot/status/audio"

    def test_init_opens_zenoh_session(self, connector, mock_providers):
        """Test initialization opens a Zenoh session."""
        assert connector.session == mock_providers["session"]
        assert connector.audio_pub == mock_providers["audio_pub"]

    def test_init_handles_zenoh_failure(self, mock_providers, make_connector):
        """Test initialization handles Zenoh session failure gracefully."""
        mock_providers["zenoh"].side_effect = Exception("Connection failed")
        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector = make_connector()
        assert connector.session is None
        assert connector.audio_pub is None

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
    async def test_connect_publishes_audio_status_via_zenoh(
        self, connector, greeting_input, mock_providers
    ):
        """Test connect publishes AudioStatus via Zenoh with UUID tracking."""
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        mock_providers["tts"].create_pending_message.return_value = {
            "text": "Hello! Nice to meet you."
        }
        await connector.connect(greeting_input)
        mock_providers["tts"].create_pending_message.assert_called_once_with(
            "Hello! Nice to meet you."
        )
        mock_providers["audio_pub"].put.assert_called_once()
        assert connector.tts_playing is True
        assert connector.tts_request_id is not None

    @pytest.mark.asyncio
    async def test_connect_does_not_set_tts_playing_without_audio_pub(
        self, connector, greeting_input, mock_providers
    ):
        """Test tts_playing stays False when audio_pub is None."""
        connector.audio_pub = None
        mock_providers["state"].process_conversation.return_value = {
            "current_state": ConversationState.CONVERSING.value
        }
        await connector.connect(greeting_input)
        assert connector.tts_playing is False

    def test_on_audio_status_resets_tts_playing_on_match(
        self, connector, mock_providers
    ):
        """Test _on_audio_status resets tts_playing when UUID and READY status match."""
        connector.tts_request_id = "test-uuid-123"
        connector.tts_playing = True
        connector.pending_finished_update = False

        mock_data = Mock()
        mock_status = Mock()
        mock_status.header.frame_id = "test-uuid-123"
        mock_status.status_speaker = 1  # READY

        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.AudioStatus"
        ) as mock_audio_cls:
            mock_audio_cls.deserialize.return_value = mock_status
            mock_audio_cls.STATUS_SPEAKER.READY.value = 1
            connector._on_audio_status(mock_data)

        assert connector.tts_playing is False
        mock_providers["ctx"].update_context.assert_not_called()

    def test_on_audio_status_triggers_deferred_context_update(
        self, connector, mock_providers
    ):
        """Test _on_audio_status calls update_context when pending_finished_update is True."""
        connector.tts_request_id = "test-uuid-123"
        connector.tts_playing = True
        connector.pending_finished_update = True

        mock_data = Mock()
        mock_status = Mock()
        mock_status.header.frame_id = "test-uuid-123"
        mock_status.status_speaker = 1  # READY

        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.AudioStatus"
        ) as mock_audio_cls:
            mock_audio_cls.deserialize.return_value = mock_status
            mock_audio_cls.STATUS_SPEAKER.READY.value = 1
            connector._on_audio_status(mock_data)

        assert connector.tts_playing is False
        assert connector.pending_finished_update is False
        mock_providers["ctx"].update_context.assert_called_once_with(
            {"greeting_conversation_finished": True}
        )

    def test_on_audio_status_ignores_non_matching_uuid(self, connector, mock_providers):
        """Test _on_audio_status does not reset tts_playing for a different UUID."""
        connector.tts_request_id = "test-uuid-123"
        connector.tts_playing = True

        mock_data = Mock()
        mock_status = Mock()
        mock_status.header.frame_id = "different-uuid-456"
        mock_status.status_speaker = 1  # READY

        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.AudioStatus"
        ) as mock_audio_cls:
            mock_audio_cls.deserialize.return_value = mock_status
            mock_audio_cls.STATUS_SPEAKER.READY.value = 1
            connector._on_audio_status(mock_data)

        assert connector.tts_playing is True

    def test_on_audio_status_ignores_non_ready_status(self, connector, mock_providers):
        """Test _on_audio_status does not reset tts_playing when speaker is ACTIVE."""
        connector.tts_request_id = "test-uuid-123"
        connector.tts_playing = True

        mock_data = Mock()
        mock_status = Mock()
        mock_status.header.frame_id = "test-uuid-123"
        mock_status.status_speaker = 2  # ACTIVE

        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.AudioStatus"
        ) as mock_audio_cls:
            mock_audio_cls.deserialize.return_value = mock_status
            mock_audio_cls.STATUS_SPEAKER.READY.value = 1
            connector._on_audio_status(mock_data)

        assert connector.tts_playing is True

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
    async def test_connect_finished_defers_context_update(
        self, connector, mock_providers
    ):
        """Test connect sets pending flag when TTS is playing."""
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
        mock_providers["ctx"].update_context.assert_not_called()
        assert connector.pending_finished_update is True
        assert connector.conversation_finished_sent is True

    @pytest.mark.asyncio
    async def test_connect_finished_updates_context_when_not_playing(
        self, connector, mock_providers
    ):
        """Test connect updates context immediately when TTS is not playing."""
        connector.audio_pub = None  # No Zenoh → tts_playing stays False
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
    async def test_connect_finished_only_sets_flag_once(
        self, connector, mock_providers
    ):
        """Test connect only sets pending flag once even if called multiple times."""
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
        mock_providers["ctx"].update_context.assert_not_called()

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
        connector.tts_playing = False
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
        connector.tts_playing = True
        connector.tts_playing_start_time = time.time()
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
        connector.tts_playing = False
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
        connector.tts_playing = False
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
        connector.tts_playing = False
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
        connector.tts_playing = False
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
        connector.tts_playing = False
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

    def test_tick_tts_timeout_prevents_freeze(self, connector, mock_providers):
        """Test tick resets tts_playing after timeout to prevent system freeze
        when Zenoh READY message is lost."""
        connector.tts_playing = True
        connector.tts_playing_start_time = (
            time.time() - 31
        )  # 31s ago, exceeds 30s timeout

        mock_providers["state"].update_state_without_llm.return_value = {
            "current_state": ConversationState.CONVERSING.value,
            "confidence": {"overall": 0.5},
            "silence_duration": 31.0,
        }

        with (
            patch(
                "actions.greeting_conversation.connector.base_greeting_conversation.logging"
            ) as mock_logging,
            patch.object(connector, "sleep"),
            patch.object(connector, "publish_countdown_status"),
        ):
            connector.tick()
            mock_logging.warning.assert_called()

        assert connector.tts_playing is False
        mock_providers["state"].update_state_without_llm.assert_called_once()

    def test_stop_sets_session_none_before_close(self, connector, mock_providers):
        """Test stop() sets self.session = None before calling close() to
        prevent race condition with publish_countdown_status."""
        session_ref = connector.session
        session_none_during_close = []

        def track_session_on_close():
            session_none_during_close.append(connector.session is None)

        session_ref.close = track_session_on_close

        with patch(
            "actions.greeting_conversation.connector.base_greeting_conversation.logging"
        ):
            connector.stop()

        assert session_none_during_close == [
            True
        ], "self.session should be None when close() is called"


class TestNormalizeTTSText:
    """Test the normalize_tts_text function."""

    def test_expand_month_abbreviations(self):
        """Test that month abbreviations are expanded correctly."""
        assert normalize_tts_text("Meet me in Jan") == "Meet me in January"
        assert normalize_tts_text("Due in Feb") == "Due in February"
        assert normalize_tts_text("Start in Mar") == "Start in March"
        assert normalize_tts_text("Born in Apr") == "Born in April"
        assert normalize_tts_text("Leave in Jun") == "Leave in June"
        assert normalize_tts_text("Hot in Jul") == "Hot in July"
        assert normalize_tts_text("Summer Aug") == "Summer August"
        assert normalize_tts_text("Fall in Sep") == "Fall in September"
        assert normalize_tts_text("Fall in Sept") == "Fall in September"
        assert normalize_tts_text("Leaves in Oct") == "Leaves in October"
        assert normalize_tts_text("Cold in Nov") == "Cold in November"
        assert normalize_tts_text("Winter Dec") == "Winter December"

    def test_month_abbreviations_case_sensitive(self):
        """Test that month abbreviations are case-sensitive."""
        assert normalize_tts_text("meet me in jan") == "meet me in jan"
        assert normalize_tts_text("Meet me in Jan") == "Meet me in January"
        assert normalize_tts_text("DUE IN FEB") == "DUE IN FEB"

    def test_month_abbreviations_word_boundaries(self):
        """Test that month abbreviations only match at word boundaries."""
        assert "January" in normalize_tts_text("Jan 1st")
        assert "January" not in normalize_tts_text("Janitor")

    def test_expand_address_abbreviations(self):
        """Test that address abbreviations are expanded correctly."""
        assert normalize_tts_text("123 Main St") == "123 Main Street"
        assert normalize_tts_text("456 Park Ave") == "456 Park Avenue"
        assert normalize_tts_text("789 Sunset Blvd") == "789 Sunset Boulevard"
        assert normalize_tts_text("101 Oak Dr") == "101 Oak Drive"
        assert normalize_tts_text("202 Elm Rd") == "202 Elm Road"
        assert normalize_tts_text("303 Maple Ln") == "303 Maple Lane"
        assert normalize_tts_text("404 Court Ct") == "404 Court Court"
        assert normalize_tts_text("505 Central Pl") == "505 Central Place"
        assert normalize_tts_text("606 Garden Pkwy") == "606 Garden Parkway"
        assert normalize_tts_text("707 Pacific Hwy") == "707 Pacific Highway"

    def test_address_abbreviations_with_period(self):
        """Test that address abbreviations with periods are expanded."""
        assert normalize_tts_text("123 Main St.") == "123 Main Street"
        assert normalize_tts_text("456 Park Ave.") == "456 Park Avenue"

    def test_expand_directional_abbreviations(self):
        """Test that directional abbreviations are expanded correctly."""
        assert normalize_tts_text("N Main Street") == "North Main Street"
        assert normalize_tts_text("S Park Avenue") == "South Park Avenue"
        # Note: Blvd is also expanded to Boulevard
        assert normalize_tts_text("E Sunset Blvd") == "East Sunset Boulevard"
        assert normalize_tts_text("W Oak Drive") == "West Oak Drive"

    def test_directional_abbreviations_with_period(self):
        """Test that directional abbreviations with periods are expanded."""
        assert normalize_tts_text("N. Main Street") == "North Main Street"
        assert normalize_tts_text("S. Park Avenue") == "South Park Avenue"

    def test_directional_abbreviations_require_following_capital(self):
        """Test that directional abbreviations only match before capitalized words."""
        assert "North" in normalize_tts_text("N Main")
        assert "North" not in normalize_tts_text("heading N")

    def test_reformat_time_on_hour(self):
        """Test that times on the hour are reformatted (remove :00)."""
        assert normalize_tts_text("Meet at 11:00 a.m.") == "Meet at 11 a.m."
        assert normalize_tts_text("Starts at 3:00 p.m.") == "Starts at 3 p.m."
        assert normalize_tts_text("Opens at 9:00") == "Opens at 9"
        assert normalize_tts_text("Closes at 10:00") == "Closes at 10"

    def test_reformat_time_with_minutes(self):
        """Test that times with minutes are reformatted (space instead of colon)."""
        assert normalize_tts_text("Meet at 11:30 a.m.") == "Meet at 11 30 a.m."
        assert normalize_tts_text("Party at 8:45 p.m.") == "Party at 8 45 p.m."
        assert normalize_tts_text("Start at 2:15") == "Start at 2 15"
        assert normalize_tts_text("Ends at 6:05") == "Ends at 6 05"

    def test_preserve_non_ascii_characters(self):
        """Test that non-ASCII characters are preserved."""
        assert normalize_tts_text("Hello café") == "Hello café"
        assert normalize_tts_text("Test 你好 text") == "Test 你好 text"
        assert normalize_tts_text("Price: €100") == "Price: €100"
        assert normalize_tts_text("Naïve résumé") == "Naïve résumé"

    def test_mixed_transformations(self):
        """Test text with multiple transformation types."""
        text = "Meet at 123 Main St on Jan 15 at 3:30 p.m."
        expected = "Meet at 123 Main Street on January 15 at 3 30 p.m."
        assert normalize_tts_text(text) == expected

    def test_address_with_directions_and_time(self):
        """Test complex address with directions and time."""
        text = "Go to N Main St at 11:00"
        expected = "Go to North Main Street at 11"
        assert normalize_tts_text(text) == expected

    def test_multiple_abbreviations_same_type(self):
        """Test text with multiple abbreviations of the same type."""
        assert (
            normalize_tts_text("From Jan to Feb to Mar")
            == "From January to February to March"
        )
        assert normalize_tts_text("Main St and Oak Ave") == "Main Street and Oak Avenue"

    def test_empty_string(self):
        """Test that empty string is handled correctly."""
        assert normalize_tts_text("") == ""

    def test_text_without_abbreviations(self):
        """Test that text without abbreviations is unchanged."""
        text = "Hello world, this is a simple sentence."
        assert normalize_tts_text(text) == text

    def test_preserve_valid_ascii_punctuation(self):
        """Test that valid ASCII punctuation is preserved."""
        text = "Hello! How are you? I'm fine, thanks."
        assert normalize_tts_text(text) == text

    def test_time_patterns_in_context(self):
        """Test time patterns in realistic contexts."""
        assert (
            normalize_tts_text("The meeting is from 9:00 to 5:30 daily.")
            == "The meeting is from 9 to 5 30 daily."
        )
        assert (
            normalize_tts_text("Call between 10:15 and 12:00")
            == "Call between 10 15 and 12"
        )

    def test_all_transformations_combined(self):
        """Test text requiring all types of transformations."""
        text = "Visit N Main St in Jan at 2:30 with café latte"
        expected = "Visit North Main Street in January at 2 30 with café latte"
        assert normalize_tts_text(text) == expected

    def test_case_preservation_where_applicable(self):
        """Test that regex patterns are case-sensitive."""
        assert normalize_tts_text("jan") == "jan"  # No match (lowercase)
        assert normalize_tts_text("Jan") == "January"  # Match (proper case)
        assert normalize_tts_text("JAN") == "JAN"  # No match (uppercase)
