from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.gps import Gps, GpsConfig


class TestGpsConfig:
    """Test cases for GpsConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GpsConfig()
        assert config.serial_port is None

    def test_custom_serial_port(self):
        """Test custom serial port configuration."""
        config = GpsConfig(serial_port="/dev/ttyUSB0")
        assert config.serial_port == "/dev/ttyUSB0"

    def test_serial_port_accepts_various_formats(self):
        """Test that serial port accepts different port formats."""
        for port in ["/dev/ttyUSB0", "/dev/ttyACM0", "COM3"]:
            config = GpsConfig(serial_port=port)
            assert config.serial_port == port


class TestGps:
    """Test cases for Gps background plugin."""

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_initialization_with_valid_port(self, mock_provider_class):
        """Test initialization with a valid serial port."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = GpsConfig(serial_port="/dev/ttyUSB0")
        background = Gps(config)

        assert background.config == config
        assert background.gps_provider == mock_provider
        mock_provider_class.assert_called_once_with(serial_port="/dev/ttyUSB0")

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = GpsConfig(serial_port="/dev/ttyUSB0")
        with caplog.at_level("INFO"):
            Gps(config)

        assert "Initiated GPS Provider with serial port: /dev/ttyUSB0 in background" in caplog.text

    def test_initialization_without_port_logs_error(self, caplog):
        """Test that initialization without a port logs an error and does not create provider."""
        config = GpsConfig()
        with caplog.at_level("ERROR"):
            background = Gps(config)

        assert "GPS serial port not specified in config" in caplog.text
        assert not hasattr(background, "gps_provider")

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_inherits_from_background(self, mock_provider_class):
        """Test that Gps inherits from Background."""
        from backgrounds.base import Background

        config = GpsConfig(serial_port="/dev/ttyUSB0")
        background = Gps(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_config_type(self, mock_provider_class):
        """Test that the config is correctly typed as GpsConfig."""
        config = GpsConfig(serial_port="/dev/ttyACM0")
        background = Gps(config)

        assert type(background.config) is GpsConfig
        assert background.config.serial_port == "/dev/ttyACM0"

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_run_calls_sleep(self, mock_provider_class):
        """Test that the default run method sleeps (inherited from Background base)."""
        config = GpsConfig(serial_port="/dev/ttyUSB0")
        background = Gps(config)

        with patch.object(background, "sleep") as mock_sleep:
            background.run()
            mock_sleep.assert_called_once_with(60)

    @patch("backgrounds.plugins.gps.GpsProvider")
    def test_initialization_with_none_port(self, mock_provider_class):
        """Test that explicitly passing None for serial_port triggers error path."""
        config = GpsConfig(serial_port=None)
        background = Gps(config)

        mock_provider_class.assert_not_called()
        assert not hasattr(background, "gps_provider")
