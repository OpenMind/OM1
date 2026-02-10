from unittest.mock import patch

from src.backgrounds.plugins.rtk import Rtk, RtkConfig


class TestRtkConfig:
    def test_config_defaults(self):
        config = RtkConfig()
        assert config.serial_port is None


class TestRtk:
    @patch("src.backgrounds.plugins.rtk.RtkProvider")
    @patch("src.backgrounds.plugins.rtk.logging")
    def test_initialization_with_valid_port(self, mock_logging, mock_provider_class):
        from unittest.mock import MagicMock

        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = RtkConfig(serial_port="/dev/ttyUSB1")

        rtk_bg = Rtk(config)

        mock_provider_class.assert_called_once_with(serial_port="/dev/ttyUSB1")
        mock_logging.info.assert_called_once_with(
            "Initiated RTK Provider with serial port: /dev/ttyUSB1 in background"
        )
        assert rtk_bg.rtk is mock_provider_instance

    @patch("src.backgrounds.plugins.rtk.logging")
    def test_initialization_without_port_logs_error(self, mock_logging):
        config = RtkConfig(serial_port=None)

        rtk_bg = Rtk(config)

        mock_logging.error.assert_called_once_with(
            "RTK serial port not specified in config"
        )
        assert not hasattr(rtk_bg, "rtk")

    @patch("src.backgrounds.plugins.rtk.RtkProvider")
    @patch("src.backgrounds.plugins.rtk.logging")
    def test_initialization_with_empty_string_port(
        self, mock_logging, mock_provider_class
    ):
        from unittest.mock import MagicMock

        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = RtkConfig(serial_port="")

        rtk_bg = Rtk(config)

        mock_provider_class.assert_called_once_with(serial_port="")
        mock_logging.info.assert_called_once_with(
            "Initiated RTK Provider with serial port:  in background"
        )
        assert rtk_bg.rtk is mock_provider_instance
