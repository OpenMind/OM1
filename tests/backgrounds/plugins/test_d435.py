from unittest.mock import MagicMock, patch

import pytest

from backgrounds.base import Background, BackgroundConfig
from backgrounds.plugins.d435 import D435


class TestD435:
    """Test cases for D435 background plugin."""

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_initialization(self, mock_provider_class):
        """Test D435 background initializes provider and calls start."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()
        assert background.d435_provider == mock_provider

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
    def test_inherits_from_background(self, mock_provider_class):
        """Test that D435 inherits from Background."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_config_stored(self, mock_provider_class):
        """Test that configuration is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = D435(config)

        assert background.config == config

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_provider_start_called_on_init(self, mock_provider_class):
        """Test that provider.start() is called during initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        D435(config)

        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.d435.D435Provider")
    def test_provider_exception_propagates(self, mock_provider_class):
        """Test that provider initialization errors propagate."""
        mock_provider_class.side_effect = Exception("Zenoh connection failed")

        config = BackgroundConfig()
        with pytest.raises(Exception, match="Zenoh connection failed"):
            D435(config)
