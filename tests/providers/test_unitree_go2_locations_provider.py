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
