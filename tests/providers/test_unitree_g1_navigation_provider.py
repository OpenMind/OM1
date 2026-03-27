from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeG1NavigationProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeG1NavigationProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeG1NavigationProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UnitreeG1NavigationProvider."""
    with (
        patch(
            "providers.unitree_g1_navigation_provider.open_zenoh_session"
        ) as mock_zenoh,
        patch(
            "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
        ) as mock_tts,
    ):

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance

        yield {
            "zenoh": mock_zenoh,
            "session": mock_session,
            "publisher": mock_publisher,
            "tts": mock_tts,
            "tts_instance": mock_tts_instance,
        }


def test_initialization(mock_dependencies):
    """Test UnitreeG1NavigationProvider initialization."""
    provider = UnitreeG1NavigationProvider(
        navigation_status_topic="nav/status",
        goal_pose_topic="nav/goal",
        cancel_goal_topic="nav/cancel",
    )

    assert provider.navigation_status_topic == "nav/status"
    assert provider.goal_pose_topic == "nav/goal"
    assert provider.cancel_goal_topic == "nav/cancel"
    assert provider.running is False
    assert provider._nav_in_progress is False
    assert provider._current_destination is None
    assert provider.navigation_status == "UNKNOWN"


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    provider = UnitreeG1NavigationProvider()

    assert provider.navigation_status_topic == "navigate_to_pose/_action/status"
    assert provider.goal_pose_topic == "goal_pose"
    assert provider.cancel_goal_topic == "navigate_to_pose/_action/cancel_goal"


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeG1NavigationProvider follows singleton pattern."""
    provider1 = UnitreeG1NavigationProvider(navigation_status_topic="topic1")
    provider2 = UnitreeG1NavigationProvider(navigation_status_topic="topic2")
    assert provider1 is provider2


def test_initialization_with_zenoh_session(mock_dependencies):
    """Test that Zenoh session is created."""
    provider = UnitreeG1NavigationProvider()

    assert provider.session == mock_dependencies["session"]
    mock_dependencies["zenoh"].assert_called_once()


def test_ai_status_publisher_initialization(mock_dependencies):
    """Test AI status publisher is created."""
    provider = UnitreeG1NavigationProvider()

    mock_dependencies["session"].declare_publisher.assert_called_once_with(
        "om/ai/request"
    )
    assert provider.ai_status_pub == mock_dependencies["publisher"]


def test_start(mock_dependencies):
    """Test starting the navigation provider."""
    provider = UnitreeG1NavigationProvider()

    provider.start()

    assert provider.running is True
    mock_dependencies["session"].declare_subscriber.assert_called_once()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = UnitreeG1NavigationProvider()

    provider.start()

    mock_dependencies["session"].declare_subscriber.reset_mock()

    provider.start()

    mock_dependencies["session"].declare_subscriber.assert_not_called()


def test_tts_provider_initialization(mock_dependencies):
    """Test that TTS provider is initialized."""
    provider = UnitreeG1NavigationProvider()

    assert provider.tts_provider == mock_dependencies["tts_instance"]
    mock_dependencies["tts"].assert_called_once()


def test_subscriber_callback_registration(mock_dependencies):
    """Test that subscriber callback is registered."""
    provider = UnitreeG1NavigationProvider()

    provider.start()

    call_args = mock_dependencies["session"].declare_subscriber.call_args
    assert call_args[0][0] == provider.navigation_status_topic
    assert callable(call_args[0][1])


def test_navigation_status_callback_accepted(mock_dependencies):
    """Test callback with ACCEPTED status (code=1)."""
    provider = UnitreeG1NavigationProvider()
    provider.start()

    mock_status = MagicMock()
    mock_status.status = 1
    mock_message = MagicMock()
    mock_message.status_list = [mock_status]

    with patch(
        "providers.unitree_g1_navigation_provider.nav_msgs.Nav2Status.deserialize",
        return_value=mock_message,
    ):
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"data"
        mock_data = MagicMock()
        mock_data.payload = mock_payload
        provider.navigation_status_message_callback(mock_data)

    assert provider.navigation_status == "ACCEPTED"
    assert provider._nav_in_progress is True


def test_navigation_status_callback_succeeded(mock_dependencies):
    """Test callback with SUCCEEDED status (code=4)."""
    provider = UnitreeG1NavigationProvider()
    provider.start()

    with provider._lock:
        provider._nav_in_progress = True
        provider._current_destination = "office"

    mock_status = MagicMock()
    mock_status.status = 4
    mock_message = MagicMock()
    mock_message.status_list = [mock_status]

    with patch(
        "providers.unitree_g1_navigation_provider.nav_msgs.Nav2Status.deserialize",
        return_value=mock_message,
    ):
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"data"
        mock_data = MagicMock()
        mock_data.payload = mock_payload
        provider.navigation_status_message_callback(mock_data)

    assert provider.navigation_status == "SUCCEEDED"
    assert provider._nav_in_progress is False


def test_navigation_status_callback_aborted(mock_dependencies):
    """Test callback with ABORTED status (code=6)."""
    provider = UnitreeG1NavigationProvider()
    provider.start()

    with provider._lock:
        provider._nav_in_progress = True

    mock_status = MagicMock()
    mock_status.status = 6
    mock_message = MagicMock()
    mock_message.status_list = [mock_status]

    with patch(
        "providers.unitree_g1_navigation_provider.nav_msgs.Nav2Status.deserialize",
        return_value=mock_message,
    ):
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"data"
        mock_data = MagicMock()
        mock_data.payload = mock_payload
        provider.navigation_status_message_callback(mock_data)

    assert provider.navigation_status == "ABORTED"
    assert provider._nav_in_progress is False


def test_navigation_status_callback_empty_payload(mock_dependencies):
    """Test callback with empty payload."""
    provider = UnitreeG1NavigationProvider()

    mock_data = MagicMock()
    mock_data.payload = None
    provider.navigation_status_message_callback(mock_data)

    assert provider.navigation_status == "UNKNOWN"


def test_navigation_status_callback_empty_status_list(mock_dependencies):
    """Test callback with empty status list."""
    provider = UnitreeG1NavigationProvider()

    mock_message = MagicMock()
    mock_message.status_list = []

    with patch(
        "providers.unitree_g1_navigation_provider.nav_msgs.Nav2Status.deserialize",
        return_value=mock_message,
    ):
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"data"
        mock_data = MagicMock()
        mock_data.payload = mock_payload
        provider.navigation_status_message_callback(mock_data)

    assert provider.navigation_status == "UNKNOWN"


def test_publish_goal_pose(mock_dependencies):
    """Test publishing a goal pose."""
    provider = UnitreeG1NavigationProvider()

    mock_pose = MagicMock()
    mock_pose.serialize.return_value = b"pose_data"

    provider.publish_goal_pose(mock_pose, "bedroom")

    assert provider._current_destination == "bedroom"
    mock_dependencies["session"].put.assert_called()


def test_publish_goal_pose_no_session(mock_dependencies):
    """Test publishing goal pose without session."""
    provider = UnitreeG1NavigationProvider()
    provider.session = None

    mock_pose = MagicMock()
    provider.publish_goal_pose(mock_pose, "kitchen")

    mock_dependencies["session"].put.assert_not_called()


def test_clear_goal_pose(mock_dependencies):
    """Test clearing/canceling navigation goal."""
    provider = UnitreeG1NavigationProvider()

    with provider._lock:
        provider._nav_in_progress = True

    provider.clear_goal_pose()

    assert provider._nav_in_progress is False
    mock_dependencies["session"].put.assert_called()


def test_clear_goal_pose_no_session(mock_dependencies):
    """Test clearing goal pose without session."""
    provider = UnitreeG1NavigationProvider()
    provider.session = None

    provider.clear_goal_pose()

    mock_dependencies["session"].put.assert_not_called()


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = UnitreeG1NavigationProvider()
    provider.start()

    provider.stop()

    assert provider.running is False
    assert provider.session is None
    assert provider.ai_status_pub is None


def test_navigation_state_property(mock_dependencies):
    """Test navigation_state property."""
    provider = UnitreeG1NavigationProvider()

    with provider._lock:
        provider.navigation_status = "EXECUTING"

    assert provider.navigation_state == "EXECUTING"


def test_is_navigating_property(mock_dependencies):
    """Test is_navigating property."""
    provider = UnitreeG1NavigationProvider()

    assert provider.is_navigating is False

    with provider._lock:
        provider._nav_in_progress = True

    assert provider.is_navigating is True


def test_init_zenoh_session_error(mock_dependencies):
    """Test __init__ when opening Zenoh session raises exception."""
    mock_dependencies["zenoh"].side_effect = Exception("Connection failed")
    provider = UnitreeG1NavigationProvider()
    assert provider.session is None


def test_init_ai_status_publisher_error(mock_dependencies):
    """Test __init__ when declare_publisher raises exception."""
    mock_dependencies["session"].declare_publisher.side_effect = Exception(
        "Publisher error"
    )
    provider = UnitreeG1NavigationProvider()
    assert provider.ai_status_pub is None


def test_publish_ai_status_publisher_none(mock_dependencies):
    """Test _publish_ai_status when ai_status_pub is None."""
    provider = UnitreeG1NavigationProvider()
    provider.ai_status_pub = None
    provider._publish_ai_status(True)


def test_publish_ai_status_exception(mock_dependencies):
    """Test _publish_ai_status when put raises exception."""
    provider = UnitreeG1NavigationProvider()
    provider.ai_status_pub = MagicMock()
    provider.ai_status_pub.put.side_effect = Exception("Put failed")
    provider._publish_ai_status(False)


def test_publish_goal_pose_exception(mock_dependencies):
    """Test publish_goal_pose when session.put raises exception."""
    provider = UnitreeG1NavigationProvider()
    mock_dependencies["session"].put.side_effect = Exception("Put failed")

    mock_pose = MagicMock()
    mock_pose.serialize.return_value = b"pose_data"
    provider.publish_goal_pose(mock_pose, "office")


def test_clear_goal_pose_exception(mock_dependencies):
    """Test clear_goal_pose when session.put raises exception."""
    provider = UnitreeG1NavigationProvider()
    mock_dependencies["session"].put.side_effect = Exception("Put failed")
    provider.clear_goal_pose()


def test_start_no_session(mock_dependencies):
    """Test start when session is None."""
    mock_dependencies["zenoh"].side_effect = Exception("No session")
    provider = UnitreeG1NavigationProvider()
    provider.start()
    assert provider.running is False
