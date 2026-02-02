import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeG1LocationsProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeG1LocationsProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeG1LocationsProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UnitreeG1LocationsProvider."""
    with (
        patch("providers.unitree_g1_locations_provider.IOProvider") as mock_io,
        patch("providers.unitree_g1_locations_provider.requests") as mock_requests,
    ):

        mock_io_instance = MagicMock()
        mock_io.return_value = mock_io_instance

        yield {
            "io": mock_io,
            "io_instance": mock_io_instance,
            "requests": mock_requests,
        }


def test_initialization(mock_dependencies):
    """Test UnitreeG1LocationsProvider initialization."""
    provider = UnitreeG1LocationsProvider(
        base_url="http://localhost:5000/locations", timeout=10, refresh_interval=60
    )

    assert provider.base_url == "http://localhost:5000/locations"
    assert provider.timeout == 10
    assert provider.refresh_interval == 60
    assert provider._locations == {}
    assert provider._thread is None


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    provider = UnitreeG1LocationsProvider()

    assert provider.base_url == "http://localhost:5000/maps/locations/list"
    assert provider.timeout == 5
    assert provider.refresh_interval == 30


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeG1LocationsProvider follows singleton pattern."""
    provider1 = UnitreeG1LocationsProvider(base_url="http://localhost:5000")
    provider2 = UnitreeG1LocationsProvider(base_url="http://localhost:6000")
    assert provider1 is provider2


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = UnitreeG1LocationsProvider()

    provider.start()

    assert provider._thread is not None
    assert provider._thread.is_alive()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = UnitreeG1LocationsProvider()

    provider.start()
    first_thread = provider._thread

    # Try to start again
    provider.start()

    # Should be the same thread
    assert provider._thread is first_thread


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = UnitreeG1LocationsProvider()

    provider.start()
    assert provider._thread is not None

    provider.stop()

    # Thread should stop
    import time

    time.sleep(0.1)
    assert not provider._thread.is_alive() or provider._stop_event.is_set()


def test_fetch_success(mock_dependencies):
    """Test successful location fetch."""
    provider = UnitreeG1LocationsProvider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "location1": {"name": "Location 1", "pose": {"x": 1.0, "y": 2.0}},
        "location2": {"name": "Location 2", "pose": {"x": 3.0, "y": 4.0}},
    }
    mock_dependencies["requests"].get.return_value = mock_response

    provider._fetch()

    assert "location1" in provider._locations
    assert "location2" in provider._locations


def test_fetch_with_nested_message(mock_dependencies):
    """Test fetch with nested message JSON."""
    import json

    provider = UnitreeG1LocationsProvider()

    locations_data = {
        "home": {"name": "Home", "pose": {"x": 0.0, "y": 0.0}},
        "kitchen": {"name": "Kitchen", "pose": {"x": 5.0, "y": 5.0}},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": json.dumps(locations_data)}
    mock_dependencies["requests"].get.return_value = mock_response

    provider._fetch()

    assert "home" in provider._locations
    assert "kitchen" in provider._locations


def test_fetch_http_error(mock_dependencies):
    """Test fetch with HTTP error response."""
    provider = UnitreeG1LocationsProvider()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_dependencies["requests"].get.return_value = mock_response

    # Should not raise exception, just log error
    provider._fetch()

    assert provider._locations == {}


def test_fetch_request_exception(mock_dependencies):
    """Test fetch with request exception."""
    provider = UnitreeG1LocationsProvider()

    mock_dependencies["requests"].get.side_effect = requests.RequestException(
        "Connection error"
    )

    provider._fetch()

    assert provider._locations == {}


def test_update_locations_dict(mock_dependencies):
    """Test updating locations with dict format."""
    provider = UnitreeG1LocationsProvider()

    locations = {
        "Location1": {"name": "Location One", "pose": {}},
        "Location2": {"name": "Location Two", "pose": {}},
    }

    provider._update_locations(locations)

    assert "location1" in provider._locations
    assert "location2" in provider._locations


def test_update_locations_list(mock_dependencies):
    """Test updating locations with list format."""
    provider = UnitreeG1LocationsProvider()

    locations = [{"name": "Location1", "pose": {}}, {"label": "Location2", "pose": {}}]

    provider._update_locations(locations)

    assert "location1" in provider._locations
    assert "location2" in provider._locations


def test_get_all_locations(mock_dependencies):
    """Test getting all locations."""
    provider = UnitreeG1LocationsProvider()

    test_locations = {
        "home": {"name": "Home", "pose": {}},
        "kitchen": {"name": "Kitchen", "pose": {}},
    }

    provider._update_locations(test_locations)

    all_locations = provider.get_all_locations()

    assert all_locations == provider._locations
    assert "home" in all_locations
    assert "kitchen" in all_locations


def test_thread_safety(mock_dependencies):
    """Test thread-safe access to locations."""
    provider = UnitreeG1LocationsProvider()

    def update_locations():
        provider._update_locations({"loc1": {"name": "Loc1"}})

    def get_locations():
        return provider.get_all_locations()

    threads = []
    for _ in range(5):
        t1 = threading.Thread(target=update_locations)
        t2 = threading.Thread(target=get_locations)
        threads.extend([t1, t2])

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert True
