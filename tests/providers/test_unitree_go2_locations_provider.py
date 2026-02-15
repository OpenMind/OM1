import time
from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2LocationsProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2LocationsProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2LocationsProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UnitreeGo2LocationsProvider."""
    with (
        patch("providers.unitree_go2_locations_provider.IOProvider") as mock_io,
        patch("providers.unitree_go2_locations_provider.requests") as mock_requests,
    ):

        mock_io_instance = MagicMock()
        mock_io.return_value = mock_io_instance

        yield {
            "io": mock_io,
            "io_instance": mock_io_instance,
            "requests": mock_requests,
        }


def test_initialization(mock_dependencies):
    """Test UnitreeGo2LocationsProvider initialization."""
    provider = UnitreeGo2LocationsProvider(
        base_url="http://localhost:5000/locations", timeout=10, refresh_interval=60
    )

    assert provider.base_url == "http://localhost:5000/locations"
    assert provider.timeout == 10
    assert provider.refresh_interval == 60
    assert provider._locations == {}


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    provider = UnitreeGo2LocationsProvider()

    assert provider.base_url == "http://localhost:5000/maps/locations/list"
    assert provider.timeout == 5
    assert provider.refresh_interval == 30


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeGo2LocationsProvider follows singleton pattern."""
    provider1 = UnitreeGo2LocationsProvider(base_url="http://localhost:5000")
    provider2 = UnitreeGo2LocationsProvider(base_url="http://localhost:6000")
    assert provider1 is provider2


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = UnitreeGo2LocationsProvider()

    provider.start()

    assert provider._thread is not None
    assert provider._thread.is_alive()


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = UnitreeGo2LocationsProvider()

    provider.start()
    provider.stop()

    time.sleep(0.1)
    assert provider._stop_event.is_set()


def test_get_all_locations_empty(mock_dependencies):
    """Test get_all_locations returns empty dict when no locations cached."""
    provider = UnitreeGo2LocationsProvider()
    result = provider.get_all_locations()
    assert result == {}


def test_get_all_locations_populated(mock_dependencies):
    """Test get_all_locations returns copy of cached locations."""
    provider = UnitreeGo2LocationsProvider()
    provider._locations = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
        "kitchen": {"name": "Kitchen", "pose": {"x": 5.0, "y": 5.0}},
    }

    result = provider.get_all_locations()
    assert "home" in result
    assert "kitchen" in result
    # Verify it's a copy (not the same object)
    assert result is not provider._locations


def test_get_location_exact_match(mock_dependencies):
    """Test get_location with exact lowercase match."""
    provider = UnitreeGo2LocationsProvider()
    provider._locations = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
    }

    result = provider.get_location("home")
    assert result == {"name": "Home", "pose": {"x": 0.0, "y": 0.0}}


def test_get_location_case_insensitive(mock_dependencies):
    """Test get_location with case-insensitive lookup."""
    provider = UnitreeGo2LocationsProvider()
    provider._locations = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
    }

    result = provider.get_location("Home")
    assert result == {"name": "Home", "pose": {"x": 0.0, "y": 0.0}}


def test_get_location_strips_whitespace(mock_dependencies):
    """Test get_location strips whitespace from label."""
    provider = UnitreeGo2LocationsProvider()
    provider._locations = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
    }

    result = provider.get_location("  home  ")
    assert result == {"name": "Home", "pose": {"x": 0.0, "y": 0.0}}


def test_get_location_missing_key(mock_dependencies):
    """Test get_location returns None for missing key."""
    provider = UnitreeGo2LocationsProvider()
    provider._locations = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
    }

    result = provider.get_location("office")
    assert result is None


def test_get_location_empty_label(mock_dependencies):
    """Test get_location returns None for empty label."""
    provider = UnitreeGo2LocationsProvider()
    result = provider.get_location("")
    assert result is None


def test_get_location_none_label(mock_dependencies):
    """Test get_location returns None for None label."""
    provider = UnitreeGo2LocationsProvider()
    result = provider.get_location(None)  # type: ignore[arg-type]
    assert result is None
