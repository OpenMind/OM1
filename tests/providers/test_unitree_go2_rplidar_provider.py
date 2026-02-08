import math
from unittest.mock import MagicMock, patch

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


class TestDistancePointToLineSegment:
    """Test cases for distance_point_to_line_segment floating-point safety."""

    @pytest.fixture
    def provider(self, mock_rplidar_dependencies):
        return UnitreeGo2RPLidarProvider()

    def test_normal_line_segment(self, provider):
        result = provider.distance_point_to_line_segment(
            px=0.0, py=1.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_point_on_line_segment(self, provider):
        result = provider.distance_point_to_line_segment(
            px=1.0, py=0.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_zero_length_segment(self, provider):
        result = provider.distance_point_to_line_segment(
            px=3.0, py=4.0, x1=0.0, y1=0.0, x2=0.0, y2=0.0
        )
        assert result == pytest.approx(5.0, rel=1e-9)

    def test_near_zero_segment_no_crash(self, provider):
        result = provider.distance_point_to_line_segment(
            px=1.0, py=1.0, x1=0.0, y1=0.0, x2=1e-16, y2=0.0
        )
        assert not math.isnan(result)
        assert not math.isinf(result)
        assert result == pytest.approx(math.sqrt(2), rel=0.01)

    def test_extremely_small_segment_no_crash(self, provider):
        result = provider.distance_point_to_line_segment(
            px=1.0, py=1.0, x1=0.0, y1=0.0, x2=1e-200, y2=1e-200
        )
        assert not math.isnan(result)
        assert not math.isinf(result)
        expected = math.sqrt(1.0**2 + 1.0**2)
        assert result == pytest.approx(expected, rel=0.01)

    def test_diagonal_line_segment(self, provider):
        result = provider.distance_point_to_line_segment(
            px=0.0, py=1.0, x1=0.0, y1=0.0, x2=1.0, y2=1.0
        )
        expected = math.sqrt(2) / 2
        assert result == pytest.approx(expected, rel=1e-6)

    def test_point_beyond_segment_end(self, provider):
        result = provider.distance_point_to_line_segment(
            px=5.0, py=0.0, x1=0.0, y1=0.0, x2=2.0, y2=0.0
        )
        assert result == pytest.approx(3.0, rel=1e-9)


def test_log_file_initialization(mock_rplidar_dependencies):
    """Test log file initialization."""
    with patch("providers.unitree_go2_rplidar_provider.time.time") as mock_time:
        mock_time.return_value = 1234567890.123456
        provider = UnitreeGo2RPLidarProvider(log_file=True)

        assert provider.write_to_local_file is True
        assert provider.filename_current == "dump/lidar_1234567890_123456Z.jsonl"
        mock_time.assert_called()
