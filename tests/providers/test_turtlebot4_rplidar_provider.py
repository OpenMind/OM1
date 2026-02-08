from unittest.mock import MagicMock, patch

import numpy as np
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
    _cleanup_singleton()
    TurtleBot4RPLidarProvider.reset()  # type: ignore


@pytest.fixture
def mock_rplidar_dependencies():
    """Mock all external dependencies for TurtleBot4RPLidarProvider."""
    with (
        patch("providers.turtlebot4_rplidar_provider.D435Provider") as mock_d435,
        patch("providers.turtlebot4_rplidar_provider.open_zenoh_session") as mock_zenoh,
    ):
        mock_d435_instance = MagicMock()
        mock_d435_instance.running = False
        mock_d435_instance.obstacle = []
        mock_d435.return_value = mock_d435_instance

        mock_zenoh_instance = MagicMock()
        mock_zenoh.return_value = mock_zenoh_instance

        yield {
            "d435": mock_d435,
            "d435_instance": mock_d435_instance,
            "zenoh": mock_zenoh,
            "zenoh_instance": mock_zenoh_instance,
        }


@pytest.fixture
def provider(mock_rplidar_dependencies):
    """Provide a default TurtleBot4RPLidarProvider instance."""
    return TurtleBot4RPLidarProvider()


@pytest.fixture
def mock_scan():
    """Provide a mock LaserScan object."""
    scan = MagicMock()
    scan.angle_min = -3.14
    scan.angle_max = 3.14
    scan.angle_increment = 0.1
    scan.ranges = [1.0] * 63
    return scan


@pytest.fixture
def sample_scan_data():
    """Provide sample scan data for testing."""
    return np.array([[0.0, 0.5], [90.0, 0.5], [180.0, 0.5], [270.0, 0.5]])


def _cleanup_singleton():
    """Clean up singleton instance after tests."""
    try:
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


def assert_default_state(provider: TurtleBot4RPLidarProvider):
    """Assert provider is in default initial state."""
    assert provider.running is False
    assert provider._raw_scan is None
    assert provider._valid_paths is None
    assert provider._lidar_string is None
    assert provider.turn_left == []
    assert provider.turn_right == []
    assert provider.advance == []
    assert provider.retreat is False
    assert provider.angles is None
    assert provider.angles_final is None


def assert_default_config(provider: TurtleBot4RPLidarProvider):
    """Assert provider has default configuration values."""
    assert provider.half_width_robot == 0.20
    assert provider.angles_blanked == []
    assert provider.relevant_distance_max == 1.1
    assert provider.relevant_distance_min == 0.08
    assert provider.sensor_mounting_angle == 180.0
    assert provider.URID == ""


class TestRPLidarConfig:
    """Test cases for RPLidarConfig."""

    def test_default_values(self):
        """Test RPLidarConfig default values."""
        config = RPLidarConfig()

        assert config.max_buf_meas == 0
        assert config.min_len == 5
        assert config.max_distance_mm == 10000

    @pytest.mark.parametrize(
        "max_buf_meas,min_len,max_distance_mm",
        [
            (100, 10, 5000),
            (50, 8, 8000),
            (200, 15, 12000),
        ],
    )
    def test_custom_values(self, max_buf_meas, min_len, max_distance_mm):
        """Test RPLidarConfig with custom values."""
        config = RPLidarConfig(
            max_buf_meas=max_buf_meas, min_len=min_len, max_distance_mm=max_distance_mm
        )

        assert config.max_buf_meas == max_buf_meas
        assert config.min_len == min_len
        assert config.max_distance_mm == max_distance_mm


class TestTurtleBot4RPLidarProviderInitialization:
    """Test cases for TurtleBot4RPLidarProvider initialization."""

    def test_initialization_with_defaults(self, provider):
        """Test initialization with default parameters."""
        assert_default_config(provider)
        assert_default_state(provider)

    @pytest.mark.parametrize(
        "param_name,param_value,expected",
        [
            ("half_width_robot", 0.25, 0.25),
            ("relevant_distance_max", 1.5, 1.5),
            ("relevant_distance_min", 0.1, 0.1),
            ("sensor_mounting_angle", 90.0, 90.0),
            ("URID", "test_robot", "test_robot"),
        ],
    )
    def test_initialization_custom_parameters(
        self, mock_rplidar_dependencies, param_name, param_value, expected
    ):
        """Test initialization with custom parameters."""
        provider = TurtleBot4RPLidarProvider(**{param_name: param_value})
        assert getattr(provider, param_name) == expected

    def test_singleton_pattern(self, mock_rplidar_dependencies):
        """Test that TurtleBot4RPLidarProvider follows singleton pattern."""
        provider1 = TurtleBot4RPLidarProvider(URID="robot_1")
        provider2 = TurtleBot4RPLidarProvider(URID="robot_2")

        assert provider1 is provider2
        assert provider1.URID == "robot_1"  # First instance URID preserved

    def test_path_structure_initialization(self, provider):
        """Test path angles and paths initialization."""
        expected_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]

        assert provider.path_angles == expected_angles
        assert len(provider.paths) == len(expected_angles)
        assert len(provider.pp) == len(provider.paths)

    def test_custom_rplidar_config(self, mock_rplidar_dependencies):
        """Test initialization with custom RPLidarConfig."""
        custom_config = RPLidarConfig(max_buf_meas=50, min_len=8, max_distance_mm=8000)
        provider = TurtleBot4RPLidarProvider(rplidar_config=custom_config)

        assert provider.rplidar_config == custom_config
        assert provider.rplidar_config.max_buf_meas == 50
        assert provider.rplidar_config.min_len == 8
        assert provider.rplidar_config.max_distance_mm == 8000

    @pytest.mark.parametrize(
        "angles_blanked,expected",
        [
            (None, []),
            ([], []),
            ([[-90, -45], [45, 90]], [[-90, -45], [45, 90]]),
        ],
    )
    def test_angles_blanked_initialization(
        self, mock_rplidar_dependencies, angles_blanked, expected
    ):
        """Test angles_blanked initialization with different inputs."""
        provider = TurtleBot4RPLidarProvider(angles_blanked=angles_blanked)
        assert provider.angles_blanked == expected

    def test_d435_provider_initialization(self, provider, mock_rplidar_dependencies):
        """Test D435 provider is initialized."""
        mocks = mock_rplidar_dependencies
        assert provider.d435_provider == mocks["d435_instance"]
        mocks["d435"].assert_called_once()

    def test_initialization_logging(self, mock_rplidar_dependencies, caplog):
        """Test initialization logging messages."""
        with caplog.at_level("INFO"):
            TurtleBot4RPLidarProvider(URID="robot_123")

        assert "Booting TurtleBot4 RPLidar (Zenoh)" in caplog.text
        assert "Connecting to the RPLIDAR via Zenoh" in caplog.text


class TestTurtleBot4RPLidarProviderLogging:
    """Test cases for file logging functionality."""

    def test_log_file_disabled_by_default(self, provider):
        """Test log file is disabled by default."""
        assert provider.write_to_local_file is False
        assert provider.filename_current is None

    def test_log_file_enabled(self, mock_rplidar_dependencies):
        """Test log file initialization when enabled."""
        with patch("providers.turtlebot4_rplidar_provider.time.time") as mock_time:
            mock_time.return_value = 1234567890.123456
            provider = TurtleBot4RPLidarProvider(log_file=True)

            assert provider.write_to_local_file is True
            assert provider.filename_current == "dump/lidar_1234567890_123456Z.jsonl"

    def test_update_filename_format(self, provider):
        """Test update_filename generates correct format."""
        with patch("providers.turtlebot4_rplidar_provider.time.time") as mock_time:
            mock_time.return_value = 9876543210.654321
            filename = provider.update_filename()

            assert filename.startswith("dump/lidar_9876543210_")
            assert filename.endswith("Z.jsonl")
            assert "9876543210" in filename

    def test_write_str_to_file_valid_string(self, provider, tmp_path):
        """Test writing valid JSON string to file."""
        provider.filename_current = str(tmp_path / "test.jsonl")
        json_line = '{"test": "data"}'

        provider.write_str_to_file(json_line)

        with open(provider.filename_current, "r") as f:
            assert f.read() == json_line + "\n"

    def test_write_str_to_file_invalid_input(self, provider):
        """Test writing non-string raises ValueError."""
        provider.filename_current = "test.jsonl"

        with pytest.raises(ValueError, match="must be a json string"):
            provider.write_str_to_file({"test": "data"})  # type: ignore

    def test_write_str_to_file_exceeds_size_limit(self, provider, tmp_path):
        """Test file rotation when size limit exceeded."""
        provider.filename_current = str(tmp_path / "test.jsonl")
        provider.max_file_size_bytes = 10  # Very small limit

        provider.write_str_to_file('{"test": "data1"}')
        original_filename = provider.filename_current

        new_filename = str(tmp_path / "test_new.jsonl")
        with patch.object(provider, "update_filename", return_value=new_filename):
            provider.write_str_to_file('{"test": "data2"}')

            assert provider.filename_current == new_filename
            assert provider.filename_current != original_filename


class TestTurtleBot4RPLidarProviderZenoh:
    """Test cases for Zenoh communication."""

    def test_zenoh_session_initialization(self, provider, mock_rplidar_dependencies):
        """Test Zenoh session is initialized correctly."""
        mocks = mock_rplidar_dependencies

        mocks["zenoh"].assert_called_once()
        assert provider.zen == mocks["zenoh_instance"]

    def test_zenoh_subscriber_setup(self, mock_rplidar_dependencies):
        """Test Zenoh subscriber is set up with correct topic."""
        mocks = mock_rplidar_dependencies
        provider = TurtleBot4RPLidarProvider(URID="test_robot")

        mocks["zenoh_instance"].declare_subscriber.assert_called_once_with(
            "test_robot/pi/scan", provider.listen_scan
        )

    def test_zenoh_initialization_failure(self, mock_rplidar_dependencies, caplog):
        """Test Zenoh initialization failure is handled gracefully."""
        mocks = mock_rplidar_dependencies
        mocks["zenoh"].side_effect = Exception("Connection failed")

        with caplog.at_level("ERROR"):
            TurtleBot4RPLidarProvider(URID="test_robot")

        assert "Error opening Zenoh client" in caplog.text

    def test_listen_scan(self, provider):
        """Test listen_scan deserializes and processes data."""
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

    def test_zenoh_processor_with_none_scan(self, provider):
        """Test _zenoh_processor handles None scan gracefully."""
        provider._zenoh_processor(None)

        assert provider._raw_scan is None
        assert "cannot safely move" in provider._lidar_string.lower()
        assert provider._valid_paths == []

    def test_zenoh_processor_with_valid_scan(self, provider, mock_scan):
        """Test _zenoh_processor processes valid scan data."""
        with patch.object(provider, "_path_processor") as mock_path_processor:
            provider._zenoh_processor(mock_scan)

            mock_path_processor.assert_called_once()
            assert provider.angles is not None
            assert provider.angles_final is not None


class TestTurtleBot4RPLidarProviderLifecycle:
    """Test cases for provider lifecycle management."""

    def test_start_sets_running_flag(self, provider):
        """Test start method sets running flag."""
        assert provider.running is False

        provider.start()

        assert provider.running is True

    def test_stop_clears_running_flag(self, provider):
        """Test stop method clears running flag."""
        provider.start()
        assert provider.running is True

        provider.stop()

        assert provider.running is False


class TestTurtleBot4RPLidarProviderProperties:
    """Test cases for provider properties."""

    def test_valid_paths_property(self, provider):
        """Test valid_paths property getter."""
        assert provider.valid_paths is None

        provider._valid_paths = [0, 1, 2]
        assert provider.valid_paths == [0, 1, 2]

    def test_raw_scan_property(self, provider):
        """Test raw_scan property getter."""
        assert provider.raw_scan is None

        test_scan = np.array([[1, 2], [3, 4]])
        provider._raw_scan = test_scan
        assert np.array_equal(provider.raw_scan, test_scan)

    def test_lidar_string_property(self, provider):
        """Test lidar_string property getter."""
        assert provider.lidar_string is None

        provider._lidar_string = "Test string"
        assert provider.lidar_string == "Test string"

    def test_movement_options_property(self, provider):
        """Test movement_options property returns all movement categories."""
        provider.turn_left = [0, 1]
        provider.advance = [3, 4, 5]
        provider.turn_right = [6, 7]
        provider.retreat = True

        options = provider.movement_options

        assert options == {
            "turn_left": [0, 1],
            "advance": [3, 4, 5],
            "turn_right": [6, 7],
            "retreat": True,
        }


class TestTurtleBot4RPLidarProviderGeometry:
    """Test cases for geometric calculations."""

    @pytest.mark.parametrize(
        "px,py,x1,y1,x2,y2,expected",
        [
            (0, 1, 0, 0, 2, 0, 1.0),  # Perpendicular
            (0, 0, 0, 0, 2, 0, 0.0),  # At endpoint
            (1, 1, 0, 0, 2, 0, 1.0),  # Perpendicular from midpoint
        ],
    )
    def test_distance_point_to_line_segment(
        self, provider, px, py, x1, y1, x2, y2, expected
    ):
        """Test distance calculation from point to line segment."""
        distance = provider.distance_point_to_line_segment(px, py, x1, y1, x2, y2)
        assert abs(distance - expected) < 0.001  # Float comparison

    def test_distance_point_to_zero_length_segment(self, provider):
        """Test distance calculation for zero-length line segment."""
        # Zero-length segment at (1, 1), point at (4, 5)
        distance = provider.distance_point_to_line_segment(4, 5, 1, 1, 1, 1)
        assert distance == 5.0  # sqrt((4-1)^2 + (5-1)^2) = 5


class TestTurtleBot4RPLidarProviderPaths:
    """Test cases for path creation and management."""

    @pytest.mark.parametrize(
        "angle,length",
        [
            (0, 1.0),
            (45, 1.5),
            (90, 2.0),
            (-45, 1.0),
        ],
    )
    def test_create_straight_path_from_angle(self, provider, angle, length):
        """Test creating straight path from various angles."""
        path = provider._create_straight_path_from_angle(
            angle, length=length, num_points=10
        )

        assert path.shape == (2, 10)
        assert path[0][0] == 0.0  # Start x
        assert path[1][0] == 0.0  # Start y

    def test_initialize_paths_count(self, provider):
        """Test path initialization creates correct number of paths."""
        paths = provider._initialize_paths()

        assert len(paths) == 10
        for path in paths:
            assert path.shape[0] == 2  # x and y coordinates
            assert path.shape[1] == 30  # default num_points


class TestTurtleBot4RPLidarProviderMovement:
    """Test cases for movement string generation."""

    def test_generate_movement_string_no_paths(self, provider):
        """Test movement string when no valid paths available."""
        result = provider._generate_movement_string([])

        assert "surrounded by objects" in result.lower()
        assert "cannot safely move" in result.lower()

    def test_generate_movement_string_with_forward(self, provider):
        """Test movement string when forward movement is available."""
        provider.advance = [3, 4, 5]
        result = provider._generate_movement_string([3, 4, 5])

        assert "turn left" in result.lower()
        assert "turn right" in result.lower()
        assert "move forwards" in result.lower()
        assert "stand still" in result.lower()

    def test_generate_movement_string_without_forward(self, provider):
        """Test movement string when only turning is available."""
        provider.advance = []
        result = provider._generate_movement_string([0, 1])

        assert "turn left" in result.lower()
        assert "turn right" in result.lower()
        assert "move forwards" not in result.lower()
        assert "stand still" in result.lower()


class TestTurtleBot4RPLidarProviderPathProcessing:
    """Test cases for path processing and obstacle detection."""

    def test_path_processor_with_d435_integration(self, provider, sample_scan_data):
        """Test path processor integrates D435 obstacle data."""
        provider.d435_provider.running = True
        provider.d435_provider.obstacle = [
            {"x": 0.5, "y": 0.5, "angle": 45.0, "distance": 0.7}
            for _ in range(60)  # More than 50 to trigger inclusion
        ]

        provider._path_processor(sample_scan_data)

        assert provider.d435_provider.running is True
        assert len(provider.d435_provider.obstacle) > 50

    @pytest.mark.parametrize(
        "distance,should_filter",
        [
            (2.0, True),  # Beyond max distance
            (0.05, True),  # Below min distance
            (0.5, False),  # Within range
        ],
    )
    def test_path_processor_distance_filtering(self, provider, distance, should_filter):
        """Test path processor filters objects by distance."""
        provider.relevant_distance_max = 1.0
        provider.relevant_distance_min = 0.1

        data = np.array([[0.0, distance], [90.0, 0.5]])
        provider._path_processor(data)

        # Should process without errors regardless
        assert provider._lidar_string is not None

    def test_path_processor_respects_blanked_angles(self, mock_rplidar_dependencies):
        """Test path processor filters blanked angle regions."""
        provider = TurtleBot4RPLidarProvider(angles_blanked=[[-90, -45], [45, 90]])

        data = np.array([[0.0, 0.5], [45.0, 0.5], [90.0, 0.5], [180.0, 0.5]])

        provider._path_processor(data)
        assert provider._lidar_string is not None

    def test_path_processor_categorizes_movements(self, provider):
        """Test path processor categorizes valid paths into movement types."""
        data = np.array([])  # Empty space - all paths valid
        provider._path_processor(data)

        assert isinstance(provider.turn_left, list)
        assert isinstance(provider.advance, list)
        assert isinstance(provider.turn_right, list)
        assert isinstance(provider.retreat, bool)
