from unittest.mock import MagicMock, patch

import pytest

from providers.rplidar_provider import RPLidarConfig, RPLidarProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    RPLidarProvider.reset()  # type: ignore
    yield
    RPLidarProvider.reset()  # type: ignore


@pytest.fixture
def mock_multiprocessing():
    with (
        patch("providers.rplidar_provider.mp.Queue") as mock_queue,
        patch("providers.rplidar_provider.mp.Process") as mock_process,
        patch("providers.rplidar_provider.threading.Thread") as mock_thread,
    ):
        mock_queue_instance = MagicMock()
        mock_process_instance = MagicMock()
        mock_thread_instance = MagicMock()
        mock_queue.return_value = mock_queue_instance
        mock_process.return_value = mock_process_instance
        mock_thread.return_value = mock_thread_instance
        yield mock_queue, mock_queue_instance, mock_process, mock_process_instance


def test_rplidar_config():
    config = RPLidarConfig()

    assert config.max_buf_meas == 0
    assert config.min_len == 5
    assert config.max_distance_mm == 10000


def test_rplidar_config_custom():
    config = RPLidarConfig(max_buf_meas=100, min_len=10, max_distance_mm=5000)

    assert config.max_buf_meas == 100
    assert config.min_len == 10
    assert config.max_distance_mm == 5000


def test_initialization(mock_multiprocessing):
    """Test RPLidarProvider initialization."""
    provider = RPLidarProvider(serial_port="/dev/ttyUSB0")

    assert provider.serial_port == "/dev/ttyUSB0"
    assert provider.running is False


def test_singleton_pattern(mock_multiprocessing):
    """Test that RPLidarProvider follows the singleton pattern."""
    provider1 = RPLidarProvider(serial_port="/dev/ttyUSB0")
    provider2 = RPLidarProvider(serial_port="/dev/ttyUSB1")
    assert provider1 is provider2


def test_start(mock_multiprocessing):
    """Test starting the RPLidarProvider."""
    _, _, _, mock_process_instance = mock_multiprocessing

    provider = RPLidarProvider(serial_port="/dev/ttyUSB0")
    provider.start()

    assert provider.running is True
    mock_process_instance.start.assert_called_once()


def test_stop(mock_multiprocessing):
    """Test stopping the RPLidarProvider."""
    _, _, _, mock_process_instance = mock_multiprocessing

    provider = RPLidarProvider(serial_port="/dev/ttyUSB0")
    provider.start()
    provider.stop()

    assert provider.running is False
    mock_process_instance.join.assert_called()
