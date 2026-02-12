from unittest.mock import MagicMock, patch

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.d435 import D435


class TestD435:
    """Test cases for D435 background plugin."""

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        assert background.config is config
        assert background.d435_provider == mock_provider
        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        with caplog.at_level("INFO"):
            D435(config)

        assert "Initiated D435 Provider in background" in caplog.text

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_provider_attribute(self, mock_provider_class):
        """Test that d435_provider attribute is set correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        assert background.d435_provider is mock_provider

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        assert background.config is config
