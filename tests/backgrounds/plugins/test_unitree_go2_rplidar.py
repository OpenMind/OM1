from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_go2_rplidar import (
    UnitreeGo2RPLidar,
    UnitreeGo2RPLidarConfig,
)


class TestUnitreeGo2RPLidarConfig:
    """Test cases for UnitreeGo2RPLidarConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2RPLidarConfig()
        assert config.serial_port is None
        assert config.half_width_robot == 0.20
        assert config.angles_blanked == []
        assert config.relevant_distance_max == 1.1
        assert config.relevant_distance_min == 0.08
        assert config.sensor_mounting_angle == 180.0
        assert config.log_file is False

    def test_custom_serial_port(self):
        """Test custom serial port configuration."""
        config = UnitreeGo2RPLidarConfig(serial_port="/dev/ttyUSB0")
        assert config.serial_port == "/dev/ttyUSB0"

    def test_custom_half_width_robot(self):
        """Test custom half width robot configuration."""
        config = UnitreeGo2RPLidarConfig(half_width_robot=0.25)
        assert config.half_width_robot == 0.25

    def test_custom_angles_blanked(self):
        """Test custom angles blanked configuration."""
        config = UnitreeGo2RPLidarConfig(angles_blanked=[0.0, 90.0, 180.0])
        assert config.angles_blanked == [0.0, 90.0, 180.0]

    def test_custom_distance_values(self):
        """Test custom distance configuration values."""
        config = UnitreeGo2RPLidarConfig(
            relevant_distance_max=2.0, relevant_distance_min=0.1
        )
        assert config.relevant_distance_max == 2.0
        assert config.relevant_distance_min == 0.1

    def test_custom_sensor_mounting_angle(self):
        """Test custom sensor mounting angle configuration."""
        config = UnitreeGo2RPLidarConfig(sensor_mounting_angle=90.0)
        assert config.sensor_mounting_angle == 90.0

    def test_custom_log_file(self):
        """Test custom log file configuration."""
        config = UnitreeGo2RPLidarConfig(log_file=True)
        assert config.log_file is True

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = UnitreeGo2RPLidarConfig(
            serial_port="/dev/ttyUSB1",
            half_width_robot=0.3,
            angles_blanked=[45.0, 135.0],
            relevant_distance_max=1.5,
            relevant_distance_min=0.15,
            sensor_mounting_angle=270.0,
            log_file=True,
        )
        assert config.serial_port == "/dev/ttyUSB1"
        assert config.half_width_robot == 0.3
        assert config.angles_blanked == [45.0, 135.0]
        assert config.relevant_distance_max == 1.5
        assert config.relevant_distance_min == 0.15
        assert config.sensor_mounting_angle == 270.0
        assert config.log_file is True


class TestUnitreeGo2RPLidar:
    """Test cases for UnitreeGo2RPLidar."""

    def test_initialization(self):
        """Test background initialization."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig(serial_port="/dev/ttyUSB0")
            background = UnitreeGo2RPLidar(config)

            assert background.config == config
            assert background.lidar_provider == mock_provider
            mock_provider.start.assert_called_once()

    def test_initialization_with_default_config(self):
        """Test background initialization with default configuration."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig()
            background = UnitreeGo2RPLidar(config)

            assert background.config == config
            assert background.lidar_provider == mock_provider
            mock_provider.start.assert_called_once()

    def test_provider_initialization_with_correct_parameters(self):
        """Test that provider is initialized with correct parameters."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig(
                serial_port="/dev/ttyUSB0",
                half_width_robot=0.25,
                angles_blanked=[0.0, 180.0],
                relevant_distance_max=2.0,
                relevant_distance_min=0.1,
                sensor_mounting_angle=90.0,
                log_file=True,
            )
            UnitreeGo2RPLidar(config)
            mock_provider_class.assert_called_once_with(
                serial_port="/dev/ttyUSB0",
                half_width_robot=0.25,
                angles_blanked=[0.0, 180.0],
                relevant_distance_max=2.0,
                relevant_distance_min=0.1,
                sensor_mounting_angle=90.0,
                log_file=True,
            )

    def test_initialization_logging(self, caplog):
        """Test that initialization logs the correct message."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig()
            with caplog.at_level("INFO"):
                UnitreeGo2RPLidar(config)

            assert "Initiated RPLidar Provider in background" in caplog.text

    def test_provider_start_is_called(self):
        """Test that provider.start() is called during initialization."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig()
            UnitreeGo2RPLidar(config)

            # Verify start was called exactly once
            mock_provider.start.assert_called_once()

    def test_config_stored_correctly(self):
        """Test that config is stored correctly in the background instance."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig(serial_port="/dev/ttyUSB0")
            background = UnitreeGo2RPLidar(config)

            assert background.config is config
            assert background.config.serial_port == "/dev/ttyUSB0"

    def test_multiple_instances_with_different_configs(self):
        """Test that multiple instances can be created with different configs."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider1 = MagicMock()
            mock_provider2 = MagicMock()
            mock_provider_class.side_effect = [mock_provider1, mock_provider2]

            config1 = UnitreeGo2RPLidarConfig(serial_port="/dev/ttyUSB0")
            config2 = UnitreeGo2RPLidarConfig(serial_port="/dev/ttyUSB1")

            background1 = UnitreeGo2RPLidar(config1)
            background2 = UnitreeGo2RPLidar(config2)

            assert background1.config.serial_port == "/dev/ttyUSB0"
            assert background2.config.serial_port == "/dev/ttyUSB1"
            assert background1.lidar_provider == mock_provider1
            assert background2.lidar_provider == mock_provider2

            assert mock_provider_class.call_count == 2
            mock_provider1.start.assert_called_once()
            mock_provider2.start.assert_called_once()

    def test_lidar_config_extraction(self):
        """Test that lidar config is extracted correctly from background config."""
        with patch(
            "backgrounds.plugins.unitree_go2_rplidar.UnitreeGo2RPLidarProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2RPLidarConfig(
                serial_port="/dev/ttyUSB2",
                half_width_robot=0.3,
                angles_blanked=[30.0, 60.0],
                relevant_distance_max=1.8,
                relevant_distance_min=0.12,
                sensor_mounting_angle=270.0,
                log_file=True,
            )
            UnitreeGo2RPLidar(config)

            # Verify the extracted config matches
            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["serial_port"] == "/dev/ttyUSB2"
            assert call_kwargs["half_width_robot"] == 0.3
            assert call_kwargs["angles_blanked"] == [30.0, 60.0]
            assert call_kwargs["relevant_distance_max"] == 1.8
            assert call_kwargs["relevant_distance_min"] == 0.12
            assert call_kwargs["sensor_mounting_angle"] == 270.0
            assert call_kwargs["log_file"] is True
