from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_state_provider import (
    UnitreeGo2StateProvider,
    state_machine_codes,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2StateProvider.reset()  # type: ignore
    yield

    try:
        provider = UnitreeGo2StateProvider()
        provider.stop()
    except Exception:
        pass

    UnitreeGo2StateProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UnitreeGo2StateProvider."""
    with (
        patch("providers.unitree_go2_state_provider.mp.Queue") as mock_queue,
        patch("providers.unitree_go2_state_provider.mp.Process") as mock_process,
        patch("providers.unitree_go2_state_provider.threading.Thread") as mock_thread,
    ):

        mock_queue_instance = MagicMock()
        mock_queue.return_value = mock_queue_instance

        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        yield {
            "queue": mock_queue,
            "queue_instance": mock_queue_instance,
            "process": mock_process,
            "process_instance": mock_process_instance,
            "thread": mock_thread,
            "thread_instance": mock_thread_instance,
        }


def test_state_machine_codes():
    """Test state machine codes dictionary."""
    assert state_machine_codes[100] == "Agile"
    assert state_machine_codes[1001] == "Damping"
    assert state_machine_codes[1007] == "Sit"
    assert state_machine_codes[1015] == "Regular Walking"
    assert state_machine_codes[2012] == "Front Flip"


def test_initialization(mock_dependencies):
    """Test UnitreeGo2StateProvider initialization."""
    provider = UnitreeGo2StateProvider(channel="test_channel")

    assert provider.channel == "test_channel"
    assert provider.state_code is None
    assert provider.state is None
    assert provider.go2_action_progress == 0


def test_initialization_default_channel(mock_dependencies):
    """Test initialization with default channel."""
    provider = UnitreeGo2StateProvider()

    assert provider.channel == ""


def test_singleton_pattern(mock_dependencies):
    """Test that UnitreeGo2StateProvider follows singleton pattern."""
    provider1 = UnitreeGo2StateProvider(channel="channel1")
    provider2 = UnitreeGo2StateProvider(channel="channel2")
    assert provider1 is provider2


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = UnitreeGo2StateProvider()

    provider.start()

    mock_dependencies["process_instance"].start.assert_called_once()
    mock_dependencies["thread_instance"].start.assert_called_once()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = UnitreeGo2StateProvider()

    # Mock the threads to appear alive
    mock_dependencies["process_instance"].is_alive.return_value = True
    mock_dependencies["thread_instance"].is_alive.return_value = True

    provider.start()

    mock_dependencies["process_instance"].start.reset_mock()
    mock_dependencies["thread_instance"].start.reset_mock()

    provider.start()

    mock_dependencies["process_instance"].start.assert_not_called()
    mock_dependencies["thread_instance"].start.assert_not_called()


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = UnitreeGo2StateProvider()

    provider.start()
    provider.stop()

    assert provider._stop_event.is_set()
    mock_dependencies["queue_instance"].put.assert_called_with("STOP")
    mock_dependencies["process_instance"].join.assert_called_once()
    mock_dependencies["thread_instance"].join.assert_called_once()


def test_state_code_initial_value(mock_dependencies):
    """Test initial state code value."""
    provider = UnitreeGo2StateProvider()

    assert provider.state_code is None


def test_state_initial_value(mock_dependencies):
    """Test initial state value."""
    provider = UnitreeGo2StateProvider()

    assert provider.state is None


def test_data_queue_creation(mock_dependencies):
    """Test that data queue is created."""
    UnitreeGo2StateProvider()

    assert mock_dependencies["queue"].call_count >= 1


def test_control_queue_creation(mock_dependencies):
    """Test that control queue is created."""
    UnitreeGo2StateProvider()

    assert mock_dependencies["queue"].call_count >= 2


def test_process_creation(mock_dependencies):
    """Test that process is created on start."""
    provider = UnitreeGo2StateProvider()

    mock_dependencies["process"].assert_not_called()

    provider.start()

    mock_dependencies["process"].assert_called_once()


def test_channel_parameter(mock_dependencies):
    """Test channel parameter is stored."""
    custom_channel = "custom/state/channel"
    provider = UnitreeGo2StateProvider(channel=custom_channel)

    assert provider.channel == custom_channel
