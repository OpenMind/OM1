from unittest.mock import MagicMock, patch

from backgrounds.base import BackgroundConfig
from backgrounds.plugins.unitree_go2_navigation import UnitreeGo2Navigation


class TestUnitreeGo2Navigation:
    """Test cases for UnitreeGo2Navigation background plugin."""

    @patch("backgrounds.plugins.unitree_go2_navigation.UnitreeGo2NavigationProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider and starts it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2Navigation(config)

        assert background.config is config
        assert background.unitree_go2_navigation_provider == mock_provider
        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_navigation.UnitreeGo2NavigationProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        with caplog.at_level("INFO"):
            UnitreeGo2Navigation(config)

        assert (
            "Unitree Go2 Navigation Provider initialized in background" in caplog.text
        )

    @patch("backgrounds.plugins.unitree_go2_navigation.UnitreeGo2NavigationProvider")
    def test_provider_attribute(self, mock_provider_class):
        """Test that provider attribute is set correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2Navigation(config)

        assert background.unitree_go2_navigation_provider is mock_provider

    @patch("backgrounds.plugins.unitree_go2_navigation.UnitreeGo2NavigationProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = BackgroundConfig()
        background = UnitreeGo2Navigation(config)

        assert background.config is config
