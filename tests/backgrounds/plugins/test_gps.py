from unittest.mock import MagicMock, patch

from src.backgrounds.plugins.gps import Gps, GpsConfig


class TestGpsConfig:
    def test_config_defaults(self):
        config = GpsConfig()
        assert config.serial_port is None


class TestGps:
    @patch("src.backgrounds.plugins.gps.GpsProvider")
    @patch("src.backgrounds.plugins.gps.logging")
    def test_initialization_with_valid_port(self, mock_logging, mock_provider_class):
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = GpsConfig(serial_port="/dev/ttyUSB0")

        gps_bg = Gps(config)

        mock_provider_class.assert_called_once_with(serial_port="/dev/ttyUSB0")
        mock_logging.info.assert_called_once_with(
            "Initiated GPS Provider with serial port: /dev/ttyUSB0 in background"
        )
        assert gps_bg.gps_provider is mock_provider_instance

    @patch("src.backgrounds.plugins.gps.logging")
    def test_initialization_without_port_logs_error(self, mock_logging):
        config = GpsConfig(serial_port=None)

        gps_bg = Gps(config)

        mock_logging.error.assert_called_once_with(
            "GPS serial port not specified in config"
        )
        # gps_provider should not be initialized if port is None
        assert not hasattr(gps_bg, "gps_provider")

    @patch("src.backgrounds.plugins.gps.GpsProvider")
    @patch("src.backgrounds.plugins.gps.logging")
    def test_initialization_with_empty_string_port(
        self, mock_logging, mock_provider_class
    ):
        # Edge case: empty string might also be considered invalid
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = GpsConfig(serial_port="")

        gps_bg = Gps(config)

        # It still tries to initialize with an empty string
        mock_provider_class.assert_called_once_with(serial_port="")
        mock_logging.info.assert_called_once_with(
            "Initiated GPS Provider with serial port:  in background"
        )
        assert gps_bg.gps_provider is mock_provider_instance
