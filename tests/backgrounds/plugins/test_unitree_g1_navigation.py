from unittest.mock import MagicMock, patch

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.unitree_g1_navigation import UnitreeG1Navigation


class TestUnitreeG1Navigation:
    """Test cases for UnitreeG1Navigation background plugin."""

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        assert background.config is config
        assert background.unitree_g1_navigation_provider == mock_provider
        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()

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
    def test_provider_attribute(self, mock_provider_class):
        """Test that provider attribute is set correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        assert background.unitree_g1_navigation_provider is mock_provider

    @patch("backgrounds.plugins.unitree_g1_navigation.UnitreeG1NavigationProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeG1Navigation(config)

        assert background.config is config
