from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2NavigationProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2NavigationProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2NavigationProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UnitreeGo2NavigationProvider."""
    with (
        patch(
            "providers.unitree_go2_navigation_provider.open_zenoh_session"
        ) as mock_zenoh,
        patch(
            "providers.unitree_go2_navigation_provider.ElevenLabsTTSProvider"
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
    """Test UnitreeGo2NavigationProvider initialization."""
    provider = UnitreeGo2NavigationProvider(
        navigation_status_topic="nav/status",
        goal_pose_topic="nav/goal",
        cancel_goal_topic="nav/cancel",
    )

    assert provider.navigation_status_topic == "nav/status"
    assert provider.goal_pose_topic == "nav/goal"
    assert provider.cancel_goal_topic == "nav/cancel"
    assert provider.running is False
    assert provider.navigation_status == "UNKNOWN"


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    provider = UnitreeGo2NavigationProvider()

    assert provider.navigation_status_topic == "navigate_to_pose/_action/status"
    assert provider.goal_pose_topic == "goal_pose"
    assert provider.cancel_goal_topic == "navigate_to_pose/_action/cancel_goal"


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeGo2NavigationProvider follows singleton pattern."""
    provider1 = UnitreeGo2NavigationProvider(navigation_status_topic="topic1")
    provider2 = UnitreeGo2NavigationProvider(navigation_status_topic="topic2")
    assert provider1 is provider2


def test_start(mock_dependencies):
    """Test starting the navigation provider."""
    provider = UnitreeGo2NavigationProvider()

    provider.start()

    assert provider.running is True
    mock_dependencies["session"].declare_subscriber.assert_called_once()
