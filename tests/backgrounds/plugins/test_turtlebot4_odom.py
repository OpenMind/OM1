from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.turtlebot4_odom import TurtleBot4Odom, TurtleBot4OdomConfig


class TestTurtleBot4OdomConfig:
    """Test cases for TurtleBot4OdomConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TurtleBot4OdomConfig()
        assert config.URID == ""

    def test_custom_urid(self):
        """Test custom URID configuration."""
        config = TurtleBot4OdomConfig(URID="turtlebot_01")
        assert config.URID == "turtlebot_01"

    def test_urid_accepts_various_formats(self):
        """Test that URID accepts different identifier formats."""
        for urid in ["robot_1", "tb4-alpha", "TURTLE_BOT_42", ""]:
            config = TurtleBot4OdomConfig(URID=urid)
            assert config.URID == urid


class TestTurtleBot4Odom:
    """Test cases for TurtleBot4Odom background plugin."""

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider with correct URID."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = TurtleBot4OdomConfig(URID="turtlebot_01")
        background = TurtleBot4Odom(config)

        assert background.config == config
        assert background.odom_provider == mock_provider
        assert background.URID == "turtlebot_01"
        mock_provider_class.assert_called_once_with("turtlebot_01")

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_initialization_with_empty_urid(self, mock_provider_class):
        """Test initialization with default empty URID."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = TurtleBot4OdomConfig()
        background = TurtleBot4Odom(config)

        assert background.URID == ""
        mock_provider_class.assert_called_once_with("")

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = TurtleBot4OdomConfig(URID="tb4_test")
        with caplog.at_level("INFO"):
            TurtleBot4Odom(config)

        assert "Initialized TurtleBot4 Odom Provider with URID: tb4_test" in caplog.text

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_urid_stored_on_instance(self, mock_provider_class):
        """Test that URID is stored on the TurtleBot4Odom instance."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = TurtleBot4OdomConfig(URID="my_turtle")
        background = TurtleBot4Odom(config)

        assert background.URID == "my_turtle"

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_inherits_from_background(self, mock_provider_class):
        """Test that TurtleBot4Odom inherits from Background."""
        from backgrounds.base import Background

        config = TurtleBot4OdomConfig()
        background = TurtleBot4Odom(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_config_type(self, mock_provider_class):
        """Test that the config is correctly typed as TurtleBot4OdomConfig."""
        config = TurtleBot4OdomConfig(URID="typed_test")
        background = TurtleBot4Odom(config)

        assert type(background.config) is TurtleBot4OdomConfig

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_run_calls_sleep(self, mock_provider_class):
        """Test that the default run method sleeps (inherited from Background base)."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = TurtleBot4OdomConfig()
        background = TurtleBot4Odom(config)

        with patch.object(background, "sleep") as mock_sleep:
            background.run()
            mock_sleep.assert_called_once_with(60)

    @patch("backgrounds.plugins.turtlebot4_odom.TurtleBot4OdomProvider")
    def test_provider_exception_propagates(self, mock_provider_class):
        """Test that exceptions from provider initialization are propagated."""
        mock_provider_class.side_effect = ConnectionError("Cannot connect to TurtleBot4")

        config = TurtleBot4OdomConfig(URID="broken_bot")
        with pytest.raises(ConnectionError, match="Cannot connect to TurtleBot4"):
            TurtleBot4Odom(config)
