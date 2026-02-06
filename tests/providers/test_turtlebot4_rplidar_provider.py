from unittest.mock import MagicMock, patch

import pytest

from providers.turtlebot4_rplidar_provider import (
    RPLidarConfig,
    TurtleBot4RPLidarProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TurtleBot4RPLidarProvider.reset()  # type: ignore
    yield

    try:
        # Get the singleton instance if it exists
        if (
            hasattr(TurtleBot4RPLidarProvider, "_singleton_instance")
            and TurtleBot4RPLidarProvider._singleton_instance is not None  # type: ignore
        ):
            provider = TurtleBot4RPLidarProvider._singleton_instance  # type: ignore
            if hasattr(provider, "running"):
                provider.running = False
            if hasattr(provider, "zen") and provider.zen:
                try:
                    provider.zen.close()
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        TurtleBot4RPLidarProvider.reset()  # type: ignore


@pytest.fixture
def mock_rplidar_dependencies():
    """Mock all external dependencies for TurtleBot4RPLidarProvider."""
    with (
        patch("providers.turtlebot4_rplidar_provider.D435Provider") as mock_d435,
        patch("providers.turtlebot4_rplidar_provider.open_zenoh_session") as mock_zenoh,
    ):
        mock_d435_instance = MagicMock()
        mock_d435.return_value = mock_d435_instance

        mock_zenoh_instance = MagicMock()
        mock_zenoh.return_value = mock_zenoh_instance

        yield {
            "d435": mock_d435,
            "d435_instance": mock_d435_instance,
            "zenoh": mock_zenoh,
            "zenoh_instance": mock_zenoh_instance,
        }


class TestRPLidarConfig:
    """Test cases for RPLidarConfig."""

    def test_default_values(self):
        """Test RPLidarConfig default values."""
        config = RPLidarConfig()

        assert config.max_buf_meas == 0
        assert config.min_len == 5
        assert config.max_distance_mm == 10000

    def test_custom_values(self):
        """Test RPLidarConfig with custom values."""
        config = RPLidarConfig(max_buf_meas=100, min_len=10, max_distance_mm=5000)

        assert config.max_buf_meas == 100
        assert config.min_len == 10
        assert config.max_distance_mm == 5000


class TestTurtleBot4RPLidarProvider:
    """Test cases for TurtleBot4RPLidarProvider."""

    def test_initialization_with_defaults(self, mock_rplidar_dependencies):
        """Test initialization with default parameters."""
        provider = TurtleBot4RPLidarProvider()

        assert provider.half_width_robot == 0.20
        assert provider.angles_blanked == []
        assert provider.relevant_distance_max == 1.1
        assert provider.relevant_distance_min == 0.08
        assert provider.sensor_mounting_angle == 180.0
        assert provider.URID == ""
        assert provider.running is False
        assert provider._raw_scan is None
        assert provider._valid_paths is None
        assert provider._lidar_string is None

    def test_initialization_with_custom_values(self, mock_rplidar_dependencies):
        """Test initialization with custom parameters."""
        provider = TurtleBot4RPLidarProvider(
            half_width_robot=0.25,
            relevant_distance_max=1.5,
            relevant_distance_min=0.1,
            sensor_mounting_angle=90.0,
            URID="test_robot",
        )

        assert provider.half_width_robot == 0.25
        assert provider.relevant_distance_max == 1.5
        assert provider.relevant_distance_min == 0.1
        assert provider.sensor_mounting_angle == 90.0
        assert provider.URID == "test_robot"

    def test_singleton_pattern(self, mock_rplidar_dependencies):
        """Test that TurtleBot4RPLidarProvider follows singleton pattern."""
        provider1 = TurtleBot4RPLidarProvider(URID="robot_1")
        provider2 = TurtleBot4RPLidarProvider(URID="robot_2")

        assert provider1 is provider2
        # First instance URID should be preserved
        assert provider1.URID == "robot_1"

    def test_path_angles_initialization(self, mock_rplidar_dependencies):
        """Test path angles initialization."""
        provider = TurtleBot4RPLidarProvider()

        assert len(provider.path_angles) == 10
        assert provider.path_angles == [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]

    def test_paths_initialization(self, mock_rplidar_dependencies):
        """Test paths initialization."""
        provider = TurtleBot4RPLidarProvider()

        assert len(provider.paths) == len(provider.path_angles)
        assert len(provider.pp) == len(provider.paths)

    def test_angles_blanked_default(self, mock_rplidar_dependencies):
        """Test that angles_blanked defaults to empty list."""
        provider = TurtleBot4RPLidarProvider()

        assert provider.angles_blanked == []

    def test_angles_blanked_custom(self, mock_rplidar_dependencies):
        """Test angles_blanked with custom values."""
        custom_blanked = [[-90, -45], [45, 90]]
        provider = TurtleBot4RPLidarProvider(angles_blanked=custom_blanked)

        assert provider.angles_blanked == custom_blanked

    def test_log_file_initialization_disabled(self, mock_rplidar_dependencies):
        """Test log file initialization when disabled."""
        provider = TurtleBot4RPLidarProvider(log_file=False)

        assert provider.write_to_local_file is False
        assert provider.filename_current is None

    def test_log_file_initialization_enabled(self, mock_rplidar_dependencies):
        """Test log file initialization when enabled."""
        with patch("providers.turtlebot4_rplidar_provider.time.time") as mock_time:
            mock_time.return_value = 1234567890.123456
            provider = TurtleBot4RPLidarProvider(log_file=True)

            assert provider.write_to_local_file is True
            assert provider.filename_current == "dump/lidar_1234567890_123456Z.jsonl"

    def test_update_filename(self, mock_rplidar_dependencies):
        """Test update_filename method."""
        provider = TurtleBot4RPLidarProvider()

        with patch("providers.turtlebot4_rplidar_provider.time.time") as mock_time:
            mock_time.return_value = 9876543210.654321
            filename = provider.update_filename()

            assert filename.startswith("dump/lidar_9876543210_")
            assert filename.endswith("Z.jsonl")
            assert "9876543210" in filename

    def test_write_str_to_file_with_valid_string(
        self, mock_rplidar_dependencies, tmp_path
    ):
        """Test writing string to file."""
        provider = TurtleBot4RPLidarProvider()
        provider.filename_current = str(tmp_path / "test.jsonl")

        json_line = '{"test": "data"}'
        provider.write_str_to_file(json_line)

        with open(provider.filename_current, "r") as f:
            content = f.read()
            assert content == json_line + "\n"

    def test_write_str_to_file_with_invalid_input(self, mock_rplidar_dependencies):
        """Test writing non-string raises ValueError."""
        provider = TurtleBot4RPLidarProvider()
        provider.filename_current = "test.jsonl"

        with pytest.raises(ValueError, match="must be a json string"):
            provider.write_str_to_file({"test": "data"})  # type: ignore

    def test_write_str_to_file_creates_new_file_on_size_limit(
        self, mock_rplidar_dependencies, tmp_path
    ):
        """Test creating new file when size limit is exceeded."""
        provider = TurtleBot4RPLidarProvider()
        provider.filename_current = str(tmp_path / "test.jsonl")
        provider.max_file_size_bytes = 10  # Very small limit

        provider.write_str_to_file('{"test": "data1"}')

        original_filename = provider.filename_current

        new_filename = str(tmp_path / "test_new.jsonl")
        with patch.object(provider, "update_filename", return_value=new_filename):
            provider.write_str_to_file('{"test": "data2"}')

            assert provider.filename_current == new_filename
            assert provider.filename_current != original_filename

    def test_start(self, mock_rplidar_dependencies):
        """Test start method sets running flag."""
        provider = TurtleBot4RPLidarProvider()

        assert provider.running is False

        provider.start()

        assert provider.running is True

    def test_zenoh_initialization(self, mock_rplidar_dependencies):
        """Test Zenoh session initialization."""
        mocks = mock_rplidar_dependencies
        provider = TurtleBot4RPLidarProvider(URID="test_robot")

        mocks["zenoh"].assert_called_once()
        assert provider.zen == mocks["zenoh_instance"]

        mocks["zenoh_instance"].declare_subscriber.assert_called_once_with(
            "test_robot/pi/scan", provider.listen_scan
        )

    def test_zenoh_initialization_failure(self, mock_rplidar_dependencies, caplog):
        """Test Zenoh initialization failure is handled."""
        mocks = mock_rplidar_dependencies
        mocks["zenoh"].side_effect = Exception("Connection failed")

        with caplog.at_level("ERROR"):
            TurtleBot4RPLidarProvider(URID="test_robot")

        assert "Error opening Zenoh client" in caplog.text

    def test_listen_scan(self, mock_rplidar_dependencies):
        """Test listen_scan method."""
        provider = TurtleBot4RPLidarProvider()

        mock_sample = MagicMock()
        mock_scan = MagicMock()

        with patch(
            "providers.turtlebot4_rplidar_provider.sensor_msgs.LaserScan.deserialize"
        ) as mock_deserialize:
            mock_deserialize.return_value = mock_scan

            with patch.object(provider, "_zenoh_processor") as mock_processor:
                provider.listen_scan(mock_sample)

                mock_deserialize.assert_called_once()
                mock_processor.assert_called_once_with(mock_scan)
                assert provider.scans == mock_scan

    def test_zenoh_processor_with_none_scan(self, mock_rplidar_dependencies):
        """Test _zenoh_processor with None scan."""
        provider = TurtleBot4RPLidarProvider()

        provider._zenoh_processor(None)

        assert provider._raw_scan is None
        assert provider._lidar_string is not None
        assert "cannot safely move" in provider._lidar_string.lower()
        assert provider._valid_paths == []

    def test_zenoh_processor_with_scan_data(self, mock_rplidar_dependencies):
        """Test _zenoh_processor with valid scan data."""
        provider = TurtleBot4RPLidarProvider()

        mock_scan = MagicMock()
        mock_scan.angle_min = -3.14
        mock_scan.angle_max = 3.14
        mock_scan.angle_increment = 0.1
        mock_scan.ranges = [1.0] * 63  # Should match the number of angles

        with patch.object(provider, "_path_processor") as mock_path_processor:
            provider._zenoh_processor(mock_scan)

            mock_path_processor.assert_called_once()
            assert provider.angles is not None
            assert provider.angles_final is not None

    def test_d435_provider_initialization(self, mock_rplidar_dependencies):
        """Test D435 provider is initialized."""
        mocks = mock_rplidar_dependencies
        provider = TurtleBot4RPLidarProvider()

        assert provider.d435_provider == mocks["d435_instance"]
        mocks["d435"].assert_called_once()

    def test_initial_state_variables(self, mock_rplidar_dependencies):
        """Test initial state of tracking variables."""
        provider = TurtleBot4RPLidarProvider()

        assert provider.turn_left == []
        assert provider.turn_right == []
        assert provider.advance == []
        assert provider.retreat is False
        assert provider.angles is None
        assert provider.angles_final is None

    def test_initialization_logging(self, mock_rplidar_dependencies, caplog):
        """Test initialization logging."""
        with caplog.at_level("INFO"):
            TurtleBot4RPLidarProvider(URID="robot_123")

        assert "Booting TurtleBot4 RPLidar (Zenoh)" in caplog.text
        assert "Connecting to the RPLIDAR via Zenoh" in caplog.text

    def test_custom_rplidar_config(self, mock_rplidar_dependencies):
        """Test initialization with custom RPLidarConfig."""
        custom_config = RPLidarConfig(max_buf_meas=50, min_len=8, max_distance_mm=8000)
        provider = TurtleBot4RPLidarProvider(rplidar_config=custom_config)

        assert provider.rplidar_config == custom_config
        assert provider.rplidar_config.max_buf_meas == 50
        assert provider.rplidar_config.min_len == 8
        assert provider.rplidar_config.max_distance_mm == 8000
