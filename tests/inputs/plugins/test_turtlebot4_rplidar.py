from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.turtlebot4_rplidar import RPLidarConfig, TurtleBot4RPLidar


class TestRPLidarConfig:
    """Test cases for RPLidarConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RPLidarConfig()
        assert config.half_width_robot == 0.20
        assert config.angles_blanked == []
        assert config.relevant_distance_max == 1.1
        assert config.relevant_distance_min == 0.08
        assert config.sensor_mounting_angle == 180.0
        assert config.URID == ""
        assert config.log_file is False

    def test_custom_half_width_robot(self):
        """Test custom half width robot configuration."""
        config = RPLidarConfig(half_width_robot=0.25)
        assert config.half_width_robot == 0.25

    def test_custom_angles_blanked(self):
        """Test custom angles blanked configuration."""
        config = RPLidarConfig(angles_blanked=[0.0, 90.0, 180.0])
        assert config.angles_blanked == [0.0, 90.0, 180.0]

    def test_custom_distance_values(self):
        """Test custom distance configuration values."""
        config = RPLidarConfig(relevant_distance_max=2.0, relevant_distance_min=0.1)
        assert config.relevant_distance_max == 2.0
        assert config.relevant_distance_min == 0.1

    def test_custom_sensor_mounting_angle(self):
        """Test custom sensor mounting angle configuration."""
        config = RPLidarConfig(sensor_mounting_angle=90.0)
        assert config.sensor_mounting_angle == 90.0

    def test_custom_urid(self):
        """Test custom URID configuration."""
        config = RPLidarConfig(URID="test_robot_123")
        assert config.URID == "test_robot_123"

    def test_custom_log_file(self):
        """Test custom log file configuration."""
        config = RPLidarConfig(log_file=True)
        assert config.log_file is True

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = RPLidarConfig(
            half_width_robot=0.3,
            angles_blanked=[45.0, 135.0],
            relevant_distance_max=1.5,
            relevant_distance_min=0.15,
            sensor_mounting_angle=270.0,
            URID="robot_xyz",
            log_file=True,
        )
        assert config.half_width_robot == 0.3
        assert config.angles_blanked == [45.0, 135.0]
        assert config.relevant_distance_max == 1.5
        assert config.relevant_distance_min == 0.15
        assert config.sensor_mounting_angle == 270.0
        assert config.URID == "robot_xyz"
        assert config.log_file is True


class TestTurtleBot4RPLidar:
    """Test cases for TurtleBot4RPLidar."""

    def test_initialization(self):
        """Test basic initialization."""
        with (
            patch(
                "inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            assert sensor.messages == []
            assert sensor.lidar == mock_provider
            mock_provider.start.assert_called_once()
            assert (
                "objects" in sensor.descriptor_for_LLM.lower()
                or "walls" in sensor.descriptor_for_LLM.lower()
            )

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        with (
            patch(
                "inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = RPLidarConfig(
                half_width_robot=0.25,
                URID="test_robot",
                log_file=True,
            )
            sensor = TurtleBot4RPLidar(config=config)

            assert sensor.lidar == mock_provider
            mock_provider.start.assert_called_once()

    def test_provider_initialization_with_correct_parameters(self):
        """Test that provider is initialized with correct parameters."""
        with (
            patch(
                "inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = RPLidarConfig(
                half_width_robot=0.25,
                angles_blanked=[0.0, 180.0],
                relevant_distance_max=2.0,
                relevant_distance_min=0.1,
                sensor_mounting_angle=90.0,
                URID="robot_123",
                log_file=True,
            )
            TurtleBot4RPLidar(config=config)
            mock_provider_class.assert_called_once_with(
                half_width_robot=0.25,
                angles_blanked=[0.0, 180.0],
                relevant_distance_max=2.0,
                relevant_distance_min=0.1,
                sensor_mounting_angle=90.0,
                URID="robot_123",
                log_file=True,
            )

    @pytest.mark.asyncio
    async def test_poll_with_lidar_data(self):
        """Test _poll with lidar data available."""
        with (
            patch(
                "inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.lidar_string = "Lidar scan data"
            mock_provider_class.return_value = mock_provider

            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            with patch(
                "inputs.plugins.turtlebot4_rplidar.asyncio.sleep", new=AsyncMock()
            ):
                result = await sensor._poll()

            assert result == "Lidar scan data"

    @pytest.mark.asyncio
    async def test_poll_with_no_data(self):
        """Test _poll when no lidar data available."""
        with (
            patch(
                "inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider.lidar_string = None
            mock_provider_class.return_value = mock_provider

            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            with patch(
                "inputs.plugins.turtlebot4_rplidar.asyncio.sleep", new=AsyncMock()
            ):
                result = await sensor._poll()

            assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_with_data(self):
        """Test _raw_to_text with valid data."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            raw_input = "Front: clear 2.5m, Left: obstacle 0.3m"
            message = await sensor._raw_to_text(raw_input)

            assert message is not None
            assert isinstance(message, Message)
            assert message.message == raw_input
            assert message.timestamp > 0

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none(self):
        """Test _raw_to_text with None input."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            message = await sensor._raw_to_text(None)

            assert message is None

    @pytest.mark.asyncio
    async def test_raw_to_text_appends_to_messages(self):
        """Test raw_to_text appends messages to buffer."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            raw_input = "Lidar scan data"
            await sensor.raw_to_text(raw_input)

            assert len(sensor.messages) == 1
            assert isinstance(sensor.messages[0], Message)
            assert sensor.messages[0].message == raw_input

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none_does_not_append(self):
        """Test raw_to_text with None does not append to messages."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            await sensor.raw_to_text(None)

            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_with_messages(self):
        """Test formatted_latest_buffer with messages in buffer."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch(
                "inputs.plugins.turtlebot4_rplidar.IOProvider"
            ) as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            # Add a message
            sensor.messages.append(
                Message(timestamp=123.456, message="Front: clear, Left: obstacle")
            )

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Front: clear, Left: obstacle" in result
            assert "INPUT:" in result
            assert "START" in result
            assert "END" in result
            assert len(sensor.messages) == 0  # Buffer should be cleared
            mock_io_provider.add_input.assert_called_once()

    def test_formatted_latest_buffer_with_empty_buffer(self):
        """Test formatted_latest_buffer with empty buffer."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            result = sensor.formatted_latest_buffer()

            assert result is None

    def test_formatted_latest_buffer_returns_latest_only(self):
        """Test formatted_latest_buffer returns only the latest message."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch(
                "inputs.plugins.turtlebot4_rplidar.IOProvider"
            ) as mock_io_provider_class,
        ):
            mock_io_provider = MagicMock()
            mock_io_provider_class.return_value = mock_io_provider

            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            # Add multiple messages
            sensor.messages.append(Message(timestamp=123.0, message="First scan"))
            sensor.messages.append(Message(timestamp=124.0, message="Second scan"))
            sensor.messages.append(Message(timestamp=125.0, message="Third scan"))

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Third scan" in result
            assert "First scan" not in result
            assert "Second scan" not in result

    def test_extract_lidar_config(self):
        """Test _extract_lidar_config method."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig(
                half_width_robot=0.3,
                angles_blanked=[30.0, 60.0],
                relevant_distance_max=1.8,
                relevant_distance_min=0.12,
                sensor_mounting_angle=270.0,
                URID="robot_abc",
                log_file=True,
            )
            sensor = TurtleBot4RPLidar(config=config)

            extracted_config = sensor._extract_lidar_config(config)

            assert extracted_config["half_width_robot"] == 0.3
            assert extracted_config["angles_blanked"] == [30.0, 60.0]
            assert extracted_config["relevant_distance_max"] == 1.8
            assert extracted_config["relevant_distance_min"] == 0.12
            assert extracted_config["sensor_mounting_angle"] == 270.0
            assert extracted_config["URID"] == "robot_abc"
            assert extracted_config["log_file"] is True

    def test_extract_lidar_config_with_defaults(self):
        """Test _extract_lidar_config with default values."""
        with (
            patch("inputs.plugins.turtlebot4_rplidar.TurtleBot4RPLidarProvider"),
            patch("inputs.plugins.turtlebot4_rplidar.IOProvider"),
        ):
            config = RPLidarConfig()
            sensor = TurtleBot4RPLidar(config=config)

            extracted_config = sensor._extract_lidar_config(config)

            assert extracted_config["half_width_robot"] == 0.20
            assert extracted_config["angles_blanked"] == []
            assert extracted_config["relevant_distance_max"] == 1.1
            assert extracted_config["relevant_distance_min"] == 0.08
            assert extracted_config["sensor_mounting_angle"] == 180.0
            assert extracted_config["URID"] == ""
            assert extracted_config["log_file"] is False
