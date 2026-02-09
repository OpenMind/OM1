from unittest.mock import MagicMock, patch

import pytest

from backgrounds.base import Background, BackgroundConfig
from backgrounds.plugins.unitree_g1_navigation import UnitreeG1Navigation


class TestUnitreeG1Navigation:
    """Test cases for UnitreeG1Navigation background plugin."""

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initializes provider and calls start."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()
        assert background.unitree_g1_navigation_provider == mock_provider

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        with caplog.at_level("INFO"):
            UnitreeG1Navigation(config)

        assert "Unitree G1 Navigation Provider initialized in background" in caplog.text

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_inherits_from_background(self, mock_provider_class):
        """Test that UnitreeG1Navigation inherits from Background."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        assert isinstance(background, Background)

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that configuration is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        assert background.config == config

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_provider_exception_propagates(self, mock_provider_class):
        """Test that provider initialization errors propagate."""
        mock_provider_class.side_effect = Exception("Navigation init failed")

        config = BackgroundConfig()
        with pytest.raises(Exception, match="Navigation init failed"):
            UnitreeG1Navigation(config)
