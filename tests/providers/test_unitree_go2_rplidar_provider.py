from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.unitree_go2_rplidar_provider import (
    RPLidarConfig,
    UnitreeGo2RPLidarProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2RPLidarProvider.reset()  # type: ignore
    yield

    try:
        # Get the singleton instance if it exists
        if hasattr(UnitreeGo2RPLidarProvider, "_singleton_instance") and UnitreeGo2RPLidarProvider._singleton_instance is not None:  # type: ignore
            provider = UnitreeGo2RPLidarProvider._singleton_instance  # type: ignore
            if hasattr(provider, "running"):
                provider.running = False
            if (
                hasattr(provider, "_serial_processor_thread")
                and provider._serial_processor_thread
            ):
                provider._serial_processor_thread.join(timeout=1)
            if (
                hasattr(provider, "_rplidar_processor_thread")
                and provider._rplidar_processor_thread
            ):
                try:
                    if hasattr(provider, "control_queue"):
                        provider.control_queue.put_nowait("STOP")
                except Exception:
                    pass
                provider._rplidar_processor_thread.join(timeout=1)
    except Exception:
        pass
    finally:
        UnitreeGo2RPLidarProvider.reset()  # type: ignore


@pytest.fixture
def mock_rplidar_dependencies():
    """Mock all external dependencies for UnitreeGo2RPLidarProvider."""
    with (
        patch(
            "providers.unitree_go2_rplidar_provider.UnitreeGo2OdomProvider"
        ) as mock_odom,
        patch("providers.unitree_go2_rplidar_provider.D435Provider") as mock_d435,
        patch("providers.unitree_go2_rplidar_provider.mp.Queue") as mock_queue,
        patch("providers.unitree_go2_rplidar_provider.mp.Process") as mock_process,
    ):
        mock_odom_instance = MagicMock()
        mock_odom.return_value = mock_odom_instance

        mock_d435_instance = MagicMock()
        mock_d435.return_value = mock_d435_instance

        mock_queue_instance = MagicMock()
        mock_queue.return_value = mock_queue_instance

        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        yield {
            "odom": mock_odom,
            "odom_instance": mock_odom_instance,
            "d435": mock_d435,
            "d435_instance": mock_d435_instance,
            "queue": mock_queue,
            "queue_instance": mock_queue_instance,
            "process": mock_process,
            "process_instance": mock_process_instance,
        }


def test_initialization(mock_rplidar_dependencies):
    """Test UnitreeGo2RPLidarProvider initialization."""
    provider = UnitreeGo2RPLidarProvider(
        serial_port="/dev/ttyUSB0",
        half_width_robot=0.25,
        relevant_distance_max=1.5,
        relevant_distance_min=0.1,
    )

    assert provider.serial_port == "/dev/ttyUSB0"
    assert provider.half_width_robot == 0.25
    assert provider.relevant_distance_max == 1.5
    assert provider.relevant_distance_min == 0.1
    assert provider.running is False
    assert provider._raw_scan is None
    assert provider._valid_paths is None


def test_singleton_pattern(mock_rplidar_dependencies):
    """Test that UnitreeGo2RPLidarProvider follows singleton pattern."""
    provider1 = UnitreeGo2RPLidarProvider(serial_port="/dev/ttyUSB0")
    provider2 = UnitreeGo2RPLidarProvider(serial_port="/dev/ttyUSB1")
    assert provider1 is provider2


def test_rplidar_config_defaults():
    """Test RPLidarConfig default values."""
    config = RPLidarConfig()

    assert config.max_buf_meas == 0
    assert config.min_len == 5
    assert config.max_distance_mm == 10000


def test_rplidar_config_custom():
    """Test RPLidarConfig with custom values."""
    config = RPLidarConfig(max_buf_meas=100, min_len=10, max_distance_mm=5000)

    assert config.max_buf_meas == 100
    assert config.min_len == 10
    assert config.max_distance_mm == 5000


def test_initialize_paths(mock_rplidar_dependencies):
    """Test path initialization."""
    provider = UnitreeGo2RPLidarProvider()

    assert len(provider.path_angles) == 10
    assert provider.path_angles == [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]
    assert len(provider.paths) == len(provider.path_angles)


def test_angles_blanked_default(mock_rplidar_dependencies):
    """Test that angles_blanked defaults to empty list."""
    provider = UnitreeGo2RPLidarProvider()
    assert provider.angles_blanked == []


def test_angles_blanked_custom(mock_rplidar_dependencies):
    """Test angles_blanked with custom values."""
    custom_blanked = [[-90, -45], [45, 90]]
    provider = UnitreeGo2RPLidarProvider(angles_blanked=custom_blanked)
    assert provider.angles_blanked == custom_blanked


def test_path_processor_filters_blanked_angles(mock_rplidar_dependencies):
    """Test that _path_processor excludes readings from blanked angle ranges.

    The blanked range [-170, -150] (in the converted [-180, +180] space)
    corresponds to angles behind the robot's mounting bracket. Readings
    that land inside this range should be filtered out before reaching
    the path planner.
    """
    mocks = mock_rplidar_dependencies
    mocks["d435_instance"].running = False
    mocks["d435_instance"].obstacle = []

    provider = UnitreeGo2RPLidarProvider(
        angles_blanked=[[-170.0, -150.0]],
        sensor_mounting_angle=180.0,
        relevant_distance_max=1.1,
        relevant_distance_min=0.08,
    )

    # Build scan data as (sensor_angle, distance_m) pairs.
    # After adding sensor_mounting_angle (180) and converting to [-180, +180]:
    #   sensor 0.0   -> oriented 180.0 -> converted 0.0    (outside blank)
    #   sensor 190.0 -> oriented 10.0  -> converted -170.0 (inside blank)
    #   sensor 200.0 -> oriented 20.0  -> converted -160.0 (inside blank)
    #   sensor 215.0 -> oriented 35.0  -> converted -145.0 (outside blank)
    scan_data = np.array(
        [
            [0.0, 0.5],
            [190.0, 0.5],
            [200.0, 0.5],
            [215.0, 0.5],
        ]
    )

    provider._path_processor(scan_data)

    assert provider._raw_scan is not None
    result_angles = provider._raw_scan[:, 2]

    assert len(result_angles) == 2
    assert -170.0 not in result_angles
    assert -160.0 not in result_angles


def test_path_processor_no_blanking_when_empty(mock_rplidar_dependencies):
    """Test that all valid readings pass through when angles_blanked is empty."""
    mocks = mock_rplidar_dependencies
    mocks["d435_instance"].running = False
    mocks["d435_instance"].obstacle = []

    provider = UnitreeGo2RPLidarProvider(
        angles_blanked=[],
        sensor_mounting_angle=180.0,
        relevant_distance_max=1.1,
        relevant_distance_min=0.08,
    )

    scan_data = np.array(
        [
            [0.0, 0.5],
            [90.0, 0.5],
            [180.0, 0.5],
        ]
    )

    provider._path_processor(scan_data)

    assert provider._raw_scan is not None
    assert len(provider._raw_scan) == 3


def test_log_file_initialization(mock_rplidar_dependencies):
    """Test log file initialization."""
    with patch("providers.unitree_go2_rplidar_provider.time.time") as mock_time:
        mock_time.return_value = 1234567890.123456
        provider = UnitreeGo2RPLidarProvider(log_file=True)

        assert provider.write_to_local_file is True
        assert provider.filename_current == "dump/lidar_1234567890_123456Z.jsonl"
        mock_time.assert_called()
