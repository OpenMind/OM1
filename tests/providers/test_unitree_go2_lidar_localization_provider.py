from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_lidar_localization_provider import (
    UnitreeGo2LidarLocalizationProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2LidarLocalizationProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2LidarLocalizationProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2LidarLocalizationProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    """Mock Zenoh dependencies."""
    with patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance


def test_initialization(mock_zenoh):
    """Test UnitreeGo2LidarLocalizationProvider initialization."""
    provider = UnitreeGo2LidarLocalizationProvider(
        topic="test/lidar_loc", quality_tolerance=0.85
    )

    assert provider.sub_topic == "test/lidar_loc"
    assert provider.quality_tolerance == 0.85
    assert provider.localization_pose is None
    assert provider.localization_status is False


def test_initialization_defaults(mock_zenoh):
    """Test initialization with default values."""
    provider = UnitreeGo2LidarLocalizationProvider()

    assert provider.sub_topic == "om/localization_pose"
    assert provider.quality_tolerance == 0.9


def test_singleton_pattern(mock_zenoh):
    """Test that UnitreeGo2LidarLocalizationProvider follows singleton pattern."""
    provider1 = UnitreeGo2LidarLocalizationProvider(topic="topic1")
    provider2 = UnitreeGo2LidarLocalizationProvider(topic="topic2")
    assert provider1 is provider2


def test_is_localized_property(mock_zenoh):
    """Test is_localized property."""
    provider = UnitreeGo2LidarLocalizationProvider()

    assert provider.is_localized is False

    provider.localization_status = True
    assert provider.is_localized is True


def test_pose_property(mock_zenoh):
    """Test pose property."""
    provider = UnitreeGo2LidarLocalizationProvider()

    assert provider.pose is None


def test_start(mock_zenoh):
    """Test starting the provider."""
    provider = UnitreeGo2LidarLocalizationProvider()

    provider.start()

    assert provider.running is True
