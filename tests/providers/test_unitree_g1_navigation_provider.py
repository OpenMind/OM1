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


def test_initialization_zenoh_failure():
    """Test handling of Zenoh initialization failure."""
    with (
        patch(
            "providers.unitree_g1_navigation_provider.open_zenoh_session"
        ) as mock_zenoh,
        patch("providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"),
    ):

        mock_zenoh.side_effect = Exception("Connection failed")

        provider = UnitreeG1NavigationProvider()

        assert provider.session is None


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


def test_start_without_session():
    """Test starting when session is None."""
    with (
        patch(
            "providers.unitree_g1_navigation_provider.open_zenoh_session"
        ) as mock_zenoh,
        patch("providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"),
    ):

        mock_zenoh.side_effect = Exception("Connection failed")

        provider = UnitreeG1NavigationProvider()

        # Should not raise exception, just log error
        provider.start()

        assert provider.running is False


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = UnitreeG1NavigationProvider()

    provider.start()

    # Reset mock
    mock_dependencies["session"].declare_subscriber.reset_mock()

    # Try to start again
    provider.start()

    # Should not subscribe again
    mock_dependencies["session"].declare_subscriber.assert_not_called()


def test_navigation_status_unknown_initially(mock_dependencies):
    """Test that navigation status is UNKNOWN initially."""
    provider = UnitreeG1NavigationProvider()

    assert provider.navigation_status == "UNKNOWN"


def test_tts_provider_initialization(mock_dependencies):
    """Test that TTS provider is initialized."""
    provider = UnitreeG1NavigationProvider()

    assert provider.tts_provider == mock_dependencies["tts_instance"]
    mock_dependencies["tts"].assert_called_once()


def test_nav_in_progress_flag(mock_dependencies):
    """Test navigation in progress flag."""
    provider = UnitreeG1NavigationProvider()

    assert provider._nav_in_progress is False


def test_current_destination_initial_value(mock_dependencies):
    """Test current destination is None initially."""
    provider = UnitreeG1NavigationProvider()

    assert provider._current_destination is None


def test_ai_status_topic(mock_dependencies):
    """Test AI status topic configuration."""
    provider = UnitreeG1NavigationProvider()

    assert provider.ai_status_topic == "om/ai/request"


def test_subscriber_callback_registration(mock_dependencies):
    """Test that subscriber callback is registered."""
    provider = UnitreeG1NavigationProvider()

    provider.start()

    call_args = mock_dependencies["session"].declare_subscriber.call_args
    assert call_args[0][0] == provider.navigation_status_topic
    assert callable(call_args[0][1])


def test_session_none_handling(mock_dependencies):
    """Test handling when session creation fails."""
    with (
        patch(
            "providers.unitree_g1_navigation_provider.open_zenoh_session"
        ) as mock_zenoh,
        patch("providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"),
    ):

        mock_zenoh.side_effect = Exception("Failed")

        provider = UnitreeG1NavigationProvider()

        assert provider.session is None
        assert provider.ai_status_pub is None
