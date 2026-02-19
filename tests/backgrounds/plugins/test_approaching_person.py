from unittest.mock import MagicMock, patch

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.approaching_person import ApproachingPerson
from providers.greeting_conversation_state_provider import ConversationState
from zenoh_msgs import Header, PersonGreetingStatus, String, Time


def test_initialization_success(caplog):
    """Test successful initialization of ApproachingPerson background."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        config = BackgroundConfig()

        with caplog.at_level("INFO"):
            background = ApproachingPerson(config)

        mock_zenoh_session.assert_called_once()
        mock_session.declare_subscriber.assert_called_once_with(
            "om/person_greeting", background._on_person_greeting
        )

        mock_state_provider_class.assert_called_once()
        assert background.greeting_state_provider == mock_state_provider
        assert "ApproachingPerson background task initialized." in caplog.text


def test_initialization_zenoh_error(caplog):
    """Test initialization when Zenoh session fails to open."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
    ):

        mock_zenoh_session.side_effect = Exception("Zenoh connection failed")
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        config = BackgroundConfig()

        with caplog.at_level("ERROR"):
            ApproachingPerson(config)

        assert "Error opening Zenoh session in ApproachingPerson" in caplog.text
        assert "Zenoh connection failed" in caplog.text


def test_on_person_greeting_approached(caplog):
    """Test callback when person is approaching (APPROACHED status)."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "backgrounds.plugins.approaching_person.ContextProvider"
        ) as mock_context_provider_class,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider
        mock_context_provider = MagicMock()
        mock_context_provider_class.return_value = mock_context_provider

        config = BackgroundConfig()
        background = ApproachingPerson(config)

        mock_sample = MagicMock()
        mock_payload = MagicMock()

        test_header = Header(stamp=Time(sec=0, nanosec=0), frame_id="test-frame")
        person_greeting_status = PersonGreetingStatus(
            header=test_header,
            request_id=String(data="test-request-id"),
            status=PersonGreetingStatus.STATUS.APPROACHED.value,
            message=String(data="Person is approaching"),
        )

        mock_payload.to_bytes.return_value = person_greeting_status.serialize()
        mock_sample.payload = mock_payload

        with caplog.at_level("DEBUG"):
            with patch.object(
                PersonGreetingStatus, "deserialize", return_value=person_greeting_status
            ):
                background._on_person_greeting(mock_sample)

        assert "Person is approaching. Triggering greeting mode." in caplog.text

        mock_context_provider.update_context.assert_called_once_with(
            {"approaching_detected": True}
        )

        mock_state_provider.reset_state.assert_called_once_with(
            ConversationState.ENGAGING
        )


def test_on_person_greeting_not_approached(caplog):
    """Test callback when person greeting status is not APPROACHED."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch(
            "backgrounds.plugins.approaching_person.ContextProvider"
        ) as mock_context_provider_class,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider
        mock_context_provider = MagicMock()
        mock_context_provider_class.return_value = mock_context_provider

        config = BackgroundConfig()
        background = ApproachingPerson(config)

        mock_sample = MagicMock()
        mock_payload = MagicMock()

        test_header = Header(stamp=Time(sec=0, nanosec=0), frame_id="test-frame")
        person_greeting_status = PersonGreetingStatus(
            header=test_header,
            request_id=String(data="test-request-id"),
            status=PersonGreetingStatus.STATUS.SWITCH.value,  # Not APPROACHED
            message=String(data="Switching state"),
        )

        mock_payload.to_bytes.return_value = person_greeting_status.serialize()
        mock_sample.payload = mock_payload

        with caplog.at_level("INFO"):
            with patch.object(
                PersonGreetingStatus, "deserialize", return_value=person_greeting_status
            ):
                background._on_person_greeting(mock_sample)

        mock_context_provider.update_context.assert_not_called()
        mock_state_provider.reset_state.assert_not_called()


def test_run_publishes_switch_status():
    """Test that run method publishes SWITCH status to person greeting topic."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
        patch("backgrounds.plugins.approaching_person.uuid4") as mock_uuid4,
        patch(
            "backgrounds.plugins.approaching_person.prepare_header"
        ) as mock_prepare_header,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        test_uuid = "test-uuid-1234"
        mock_uuid4.return_value = test_uuid
        test_header = Header(stamp=Time(sec=0, nanosec=0), frame_id="test-uuid-1234")
        mock_prepare_header.return_value = test_header

        config = BackgroundConfig()
        background = ApproachingPerson(config)

        with patch.object(background, "sleep") as mock_sleep:
            background.run()

        assert mock_session.put.call_count == 1
        call_args = mock_session.put.call_args

        assert call_args[0][0] == "om/person_greeting"
        published_data = call_args[0][1]
        assert isinstance(published_data, bytes)

        mock_sleep.assert_called_once_with(5)


def test_stop_closes_session(caplog):
    """Test that stop method closes the Zenoh session."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        config = BackgroundConfig()
        background = ApproachingPerson(config)

        with caplog.at_level("INFO"):
            background.stop()

        mock_session.close.assert_called_once()
        assert "Stopping ApproachingPerson background task." in caplog.text


def test_stop_without_session(caplog):
    """Test that stop method handles missing session gracefully."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
    ):

        mock_zenoh_session.side_effect = Exception("No session")
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        config = BackgroundConfig()
        background = ApproachingPerson(config)
        background.session = None  # type: ignore

        with caplog.at_level("INFO"):
            background.stop()

        assert "Stopping ApproachingPerson background task." in caplog.text


def test_person_greeting_topic_constant():
    """Test that person greeting topic is set correctly."""
    with (
        patch(
            "backgrounds.plugins.approaching_person.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "backgrounds.plugins.approaching_person.GreetingConversationStateMachineProvider"
        ) as mock_state_provider_class,
    ):

        mock_session = MagicMock()
        mock_zenoh_session.return_value = mock_session
        mock_state_provider = MagicMock()
        mock_state_provider_class.return_value = mock_state_provider

        config = BackgroundConfig()
        background = ApproachingPerson(config)

        assert background.person_greeting_topic == "om/person_greeting"
