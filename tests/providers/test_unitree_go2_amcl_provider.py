from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2AMCLProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2AMCLProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2AMCLProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    """Mock Zenoh dependencies."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance


def test_initialization(mock_zenoh):
    """Test UnitreeGo2AMCLProvider initialization."""
    provider = UnitreeGo2AMCLProvider(
        topic="test/amcl", pose_tolerance=0.5, yaw_tolerance=0.3
    )

    assert provider.sub_topic == "test/amcl"
    assert provider.pose_tolerance == 0.5
    assert provider.yaw_tolerance == 0.3
    assert provider.localization_pose is None
    assert provider.localization_status is False


def test_initialization_defaults(mock_zenoh):
    """Test initialization with default values."""
    provider = UnitreeGo2AMCLProvider()

    assert provider.sub_topic == "amcl_pose"
    assert provider.pose_tolerance == 0.4
    assert provider.yaw_tolerance == 0.2


def test_singleton_pattern(mock_zenoh):
    """Test that UnitreeGo2AMCLProvider follows singleton pattern."""
    provider1 = UnitreeGo2AMCLProvider(topic="topic1")
    provider2 = UnitreeGo2AMCLProvider(topic="topic2")
    assert provider1 is provider2


def test_is_localized_property(mock_zenoh):
    """Test is_localized property."""
    provider = UnitreeGo2AMCLProvider()

    assert provider.is_localized is False

    provider.localization_status = True
    assert provider.is_localized is True


def test_start(mock_zenoh):
    """Test starting the provider."""
    provider = UnitreeGo2AMCLProvider()

    provider.start()

    assert provider.running is True
