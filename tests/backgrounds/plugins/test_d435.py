from unittest.mock import MagicMock, patch

from src.backgrounds.base import BackgroundConfig
from src.backgrounds.plugins.d435 import D435


class TestD435:
    @patch("src.backgrounds.plugins.d435.D435Provider")
    @patch("src.backgrounds.plugins.d435.logging")
    def test_initialization_and_start(self, mock_logging, mock_provider_class):
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance
        config = BackgroundConfig()

        d435_bg = D435(config)

        mock_provider_class.assert_called_once()
        mock_provider_instance.start.assert_called_once()
        mock_logging.info.assert_called_once_with(
            "Initiated D435 Provider in background"
        )
        assert d435_bg.d435_provider is mock_provider_instance
