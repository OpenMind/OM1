from unittest.mock import MagicMock, patch

from backgrounds.plugins.rtk import Rtk, RtkConfig


class TestRtkConfig:
    """Test cases for RtkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RtkConfig()
        assert config.serial_port is None

    def test_custom_serial_port(self):
        """Test custom serial port configuration."""
        config = RtkConfig(serial_port="/dev/ttyUSB0")
        assert config.serial_port == "/dev/ttyUSB0"


class TestRtk:
    """Test cases for Rtk background plugin."""

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_initialization_with_port(self, mock_provider_class):
        """Test background initialization with valid serial port."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = RtkConfig(serial_port="/dev/ttyUSB0")
        background = Rtk(config)

        assert background.config is config
        assert background.rtk == mock_provider
        mock_provider_class.assert_called_once_with(serial_port="/dev/ttyUSB0")

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_initialization_logging_with_port(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message with port."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = RtkConfig(serial_port="/dev/ttyUSB0")
        with caplog.at_level("INFO"):
            Rtk(config)

        assert (
            "Initiated RTK Provider with serial port: /dev/ttyUSB0 in background"
            in caplog.text
        )

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_initialization_with_none_port_early_return(self, mock_provider_class):
        """Test that None port causes early return without creating provider."""
        config = RtkConfig()
        background = Rtk(config)

        mock_provider_class.assert_not_called()
        assert not hasattr(background, "rtk")

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_initialization_with_none_port_error_log(self, mock_provider_class, caplog):
        """Test that None port logs an error message."""
        config = RtkConfig()
        with caplog.at_level("ERROR"):
            Rtk(config)

        assert "RTK serial port not specified in config" in caplog.text

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_no_start_called(self, mock_provider_class):
        """Test that start() is NOT called on provider (RTK has no start)."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = RtkConfig(serial_port="/dev/ttyUSB0")
        Rtk(config)

        mock_provider.start.assert_not_called()

    @patch("backgrounds.plugins.rtk.RtkProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = RtkConfig(serial_port="/dev/ttyUSB1")
        background = Rtk(config)

        assert background.config is config
        assert background.config.serial_port == "/dev/ttyUSB1"
