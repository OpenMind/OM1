from unittest.mock import MagicMock, patch

from backgrounds.plugins.turtlebot4_odom import (
    TurtleBot4Odom,
    TurtleBot4OdomConfig,
)


class TestTurtleBot4OdomConfig:
    """Test cases for TurtleBot4OdomConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TurtleBot4OdomConfig()
        assert config.URID == ""

    def test_custom_urid(self):
        """Test custom URID configuration."""
        config = TurtleBot4OdomConfig(URID="test_robot_123")
        assert config.URID == "test_robot_123"

    def test_config_inherits_from_background_config(self):
        """Test that config properly inherits from BackgroundConfig."""
        config = TurtleBot4OdomConfig(URID="robot_456")
        assert hasattr(config, "URID")
        assert config.URID == "robot_456"


class TestTurtleBot4Odom:
    """Test cases for TurtleBot4Odom."""

    def test_initialization(self):
        """Test background initialization."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = TurtleBot4OdomConfig(URID="test_robot")
            background = TurtleBot4Odom(config)

            assert background.config == config
            assert background.URID == "test_robot"
            assert background.odom_provider == mock_provider
            mock_provider_class.assert_called_once_with("test_robot")

    def test_initialization_with_empty_urid(self):
        """Test background initialization with empty URID."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = TurtleBot4OdomConfig()
            background = TurtleBot4Odom(config)

            assert background.URID == ""
            assert background.odom_provider == mock_provider
            mock_provider_class.assert_called_once_with("")

    def test_initialization_logging(self, caplog):
        """Test that initialization logs the correct message."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = TurtleBot4OdomConfig(URID="test_robot_123")
            with caplog.at_level("INFO"):
                TurtleBot4Odom(config)

            assert (
                "Initialized TurtleBot4 Odom Provider with URID: test_robot_123"
                in caplog.text
            )

    def test_provider_initialization_with_correct_urid(self):
        """Test that provider is initialized with the correct URID."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = TurtleBot4OdomConfig(URID="unique_robot_id_789")
            background = TurtleBot4Odom(config)

            # Verify provider was called with the exact URID from config
            mock_provider_class.assert_called_once_with("unique_robot_id_789")
            assert background.URID == "unique_robot_id_789"

    def test_config_stored_correctly(self):
        """Test that config is stored correctly in the background instance."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = TurtleBot4OdomConfig(URID="robot_xyz")
            background = TurtleBot4Odom(config)

            assert background.config is config
            assert background.config.URID == "robot_xyz"

    def test_multiple_instances_with_different_urids(self):
        """Test that multiple instances can be created with different URIDs."""
        with patch(
            "backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class:
            mock_provider1 = MagicMock()
            mock_provider2 = MagicMock()
            mock_provider_class.side_effect = [mock_provider1, mock_provider2]

            config1 = TurtleBot4OdomConfig(URID="robot_1")
            config2 = TurtleBot4OdomConfig(URID="robot_2")

            background1 = TurtleBot4Odom(config1)
            background2 = TurtleBot4Odom(config2)

            assert background1.URID == "robot_1"
            assert background2.URID == "robot_2"
            assert background1.odom_provider == mock_provider1
            assert background2.odom_provider == mock_provider2

            assert mock_provider_class.call_count == 2
            mock_provider_class.assert_any_call("robot_1")
            mock_provider_class.assert_any_call("robot_2")
