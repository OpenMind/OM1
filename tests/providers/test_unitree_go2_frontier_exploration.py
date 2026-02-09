from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_frontier_exploration import (
    UnitreeGo2FrontierExplorationProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2FrontierExplorationProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2FrontierExplorationProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2FrontierExplorationProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies."""
    with (
        patch("providers.zenoh_listener_provider.open_zenoh_session") as mock_zenoh,
        patch(
            "providers.unitree_go2_frontier_exploration.ContextProvider"
        ) as mock_context,
    ):

        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance

        yield {
            "zenoh": mock_zenoh,
            "session": mock_session,
            "context": mock_context,
            "context_instance": mock_context_instance,
        }


def test_initialization(mock_dependencies):
    """Test UnitreeGo2FrontierExplorationProvider initialization."""
    provider = UnitreeGo2FrontierExplorationProvider(
        topic="test/explore", context_aware_text={"exploration_done": False}
    )

    assert provider.sub_topic == "test/explore"
    assert provider.context_aware_text == {"exploration_done": False}
    assert provider.exploration_info is None
    assert provider.exploration_complete is False


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    provider = UnitreeGo2FrontierExplorationProvider()

    assert provider.sub_topic == "explore/status"
    assert provider.context_aware_text == {"exploration_done": True}


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeGo2FrontierExplorationProvider follows singleton pattern."""
    provider1 = UnitreeGo2FrontierExplorationProvider(topic="topic1")
    provider2 = UnitreeGo2FrontierExplorationProvider(topic="topic2")
    assert provider1 is provider2


def test_status_property(mock_dependencies):
    """Test status property."""
    provider = UnitreeGo2FrontierExplorationProvider()

    assert provider.status is False

    provider.exploration_complete = True
    assert provider.status is True


def test_info_property(mock_dependencies):
    """Test info property."""
    provider = UnitreeGo2FrontierExplorationProvider()

    assert provider.info is None

    provider.exploration_info = "Exploration completed successfully"
    assert provider.info == "Exploration completed successfully"


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = UnitreeGo2FrontierExplorationProvider()

    provider.start()

    assert provider.running is True


def test_context_provider_initialization(mock_dependencies):
    """Test that ContextProvider is initialized."""
    provider = UnitreeGo2FrontierExplorationProvider()

    assert provider.context_provider == mock_dependencies["context_instance"]
    mock_dependencies["context"].assert_called_once()
